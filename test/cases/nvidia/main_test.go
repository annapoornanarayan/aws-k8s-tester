//go:build e2e

package nvidia

import (
	"context"
	_ "embed"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"slices"
	"strings"

	"testing"
	"time"

	fwext "github.com/aws/aws-k8s-tester/internal/e2e"
	"github.com/aws/aws-k8s-tester/test/manifests"
	"github.com/aws/aws-sdk-go-v2/aws"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/e2e-framework/klient/wait"
	"sigs.k8s.io/e2e-framework/klient/wait/conditions"
	"sigs.k8s.io/e2e-framework/pkg/env"
	"sigs.k8s.io/e2e-framework/pkg/envconf"
)

var (
	testenv                env.Environment
	nodeType               *string
	installDevicePlugin    *bool
	efaEnabled             *bool
	nvidiaTestImage        *string
	pytorchImage           *string
	skipUnitTestSubcommand *string
	amiVariant             *string
	kubernetesVersion      *string
	teamIdentifier         *string
	nodeCount              int
	gpuPerNode             int
	efaPerNode             int
)

func deployMPIOperator(ctx context.Context, config *envconf.Config) (context.Context, error) {
	dep := appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mpi-operator", Namespace: "mpi-operator"},
	}
	err := wait.For(conditions.New(config.Client().Resources()).DeploymentConditionMatch(&dep, appsv1.DeploymentAvailable, v1.ConditionTrue),
		wait.WithContext(ctx))
	if err != nil {
		return ctx, fmt.Errorf("failed to deploy mpi-operator: %v", err)
	}
	return ctx, nil
}

func deployNvidiaDevicePlugin(ctx context.Context, config *envconf.Config) (context.Context, error) {
	ds := appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "nvidia-device-plugin-daemonset", Namespace: "kube-system"},
	}
	err := wait.For(fwext.NewConditionExtension(config.Client().Resources()).DaemonSetReady(&ds),
		wait.WithContext(ctx))
	if err != nil {
		return ctx, fmt.Errorf("failed to deploy nvidia-device-plugin: %v", err)
	}
	return ctx, nil
}

func deployEFAPlugin(ctx context.Context, config *envconf.Config) (context.Context, error) {
	ds := appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "aws-efa-k8s-device-plugin-daemonset", Namespace: "kube-system"},
	}
	err := wait.For(fwext.NewConditionExtension(config.Client().Resources()).DaemonSetReady(&ds),
		wait.WithContext(ctx))
	if err != nil {
		return ctx, fmt.Errorf("failed to deploy efa-device-plugin: %v", err)
	}

	return ctx, nil
}

func checkNodeTypes(ctx context.Context, config *envconf.Config) (context.Context, error) {
	clientset, err := kubernetes.NewForConfig(config.Client().RESTConfig())
	if err != nil {
		return ctx, err
	}

	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return ctx, err
	}

	for i := 1; i < len(nodes.Items)-1; i++ {
		if nodes.Items[i].Labels["node.kubernetes.io/instance-type"] != nodes.Items[i-1].Labels["node.kubernetes.io/instance-type"] {
			return ctx, fmt.Errorf("Node types are not the same, all node types must be the same in the cluster")
		}
	}

	if *nodeType != "" {
		for _, v := range nodes.Items {
			if v.Labels["node.kubernetes.io/instance-type"] == *nodeType {
				nodeCount++
				gpu := v.Status.Capacity["nvidia.com/gpu"]
				gpuPerNode = int(gpu.Value())
				efa := v.Status.Capacity["vpc.amazonaws.com/efa"]
				efaPerNode = int(efa.Value())
			}
		}
	} else {
		log.Printf("No node type specified. Using the node type %s in the node groups.", nodes.Items[0].Labels["node.kubernetes.io/instance-type"])
		nodeType = aws.String(nodes.Items[0].Labels["node.kubernetes.io/instance-type"])
		nodeCount = len(nodes.Items)
		gpu := nodes.Items[0].Status.Capacity["nvidia.com/gpu"]
		gpuPerNode = int(gpu.Value())
		efa := nodes.Items[0].Status.Capacity["vpc.amazonaws.com/efa"]
		efaPerNode = int(efa.Value())
	}

	return ctx, nil
}

func getRegionFromNodes(ctx context.Context, config *envconf.Config) (string, error) {
	clientset, err := kubernetes.NewForConfig(config.Client().RESTConfig())
	if err != nil {
		return "", fmt.Errorf("failed to create Kubernetes client: %w", err)
	}
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{Limit: 1})
	if err != nil {
		return "", fmt.Errorf("failed to list nodes: %w", err)
	}
	az := nodes.Items[0].Labels["topology.kubernetes.io/zone"]
	return strings.TrimSuffix(az, az[len(az)-1:]), nil
}
func deployDaemonSet(name, namespace string) env.Func {
	return func(ctx context.Context, config *envconf.Config) (context.Context, error) {
		log.Printf("Waiting for %s daemonset to be ready.", name)
		daemonset := appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		}
		err := wait.For(
			fwext.NewConditionExtension(config.Client().Resources()).DaemonSetReady(&daemonset),
			wait.WithTimeout(5*time.Minute),
		)
		if err != nil {
			return ctx, fmt.Errorf("%s daemonset is not ready: %w", name, err)
		}
		log.Printf("%s daemonset is ready.", name)
		return ctx, nil
	}
}

func TestMain(m *testing.M) {
	nodeType = flag.String("nodeType", "", "node type for the tests")
	nvidiaTestImage = flag.String("nvidiaTestImage", "", "nccl test image for nccl tests")
	pytorchImage = flag.String("pytorchImage", "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-ec2", "pytorch cuda image for single node tests")
	efaEnabled = flag.Bool("efaEnabled", false, "enable efa tests")
	installDevicePlugin = flag.Bool("installDevicePlugin", true, "install nvidia device plugin")
	skipUnitTestSubcommand = flag.String("skipUnitTestSubcommand", "", "optional command to skip specified unit test, `-s test1|test2|...`")
	amiVariant = flag.String("amiVariant", "al2", "AMI variant (al2 or al2023)")
	kubernetesVersion = flag.String("kubernetesVersion", "1.32", "Kubernetes version (1.32 or 1.33)")
	teamIdentifier = flag.String("teamIdentifier", "node-runtime", "Team identifier for CloudWatch metrics")

	cfg, err := envconf.NewFromFlags()
	if err != nil {
		log.Fatalf("failed to initialize test environment: %v", err)
	}

	// Validate and transform flags
	*amiVariant = strings.ToLower(*amiVariant)
	if *kubernetesVersion != "1.32" && *kubernetesVersion != "1.33" {
		log.Fatalf("kubernetesVersion must be either 1.32 or 1.33, got: %s", *kubernetesVersion)
	}
	manifestVersion := strings.Replace(*kubernetesVersion, ".", "", 1)
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	testenv = env.NewWithConfig(cfg).WithContext(ctx)
	region, err := getRegionFromNodes(ctx, cfg)
	if err != nil {
		log.Printf("Warning: failed to get region from nodes, using default us-west-2: %v", err)
		region = "us-west-2"
	}
	// template data for CloudWatch Agent manifest
	templateData := map[string]string{
		"VERSION":         manifestVersion,
		"VARIANT":         *amiVariant,
		"INSTANCE_TYPE":   *nodeType,
		"REGION":          region,
		"TEAM_IDENTIFIER": *teamIdentifier,
		"TEST_NAME":       "nvidia",
	}
	renderedCloudWatchAgentManifest, err := fwext.RenderManifests(manifests.CloudWatchAgentManifest, templateData)
	if err != nil {
		log.Fatalf("failed to render CloudWatch Agent manifest: %v", err)
	}
	// all NVIDIA tests require the device plugin and MPI operator
	deploymentManifests := [][]byte{
		manifests.MpiOperatorManifest,
		manifests.DCGMExporterManifest,
		renderedCloudWatchAgentManifest,
	}
	setUpFunctions := []env.Func{
		func(ctx context.Context, config *envconf.Config) (context.Context, error) {
			log.Println("Applying MPI operator, DCGM Exporter and CloudWatch Agent manifests.")
			err := fwext.ApplyManifests(config.Client().RESTConfig(), deploymentManifests...)
			if err != nil {
				return ctx, err
			}
			log.Println("Successfully applied MPI operator, DCGM Exporter and CloudWatch Agent manifests.")
			return ctx, nil
		},
		deployMPIOperator,
		deployDaemonSet("dcgm-exporter", "kube-system"),
		deployDaemonSet("cwagent-prometheus", "amazon-cloudwatch"),
	}

	if *installDevicePlugin {
		deploymentManifests = append(deploymentManifests, manifests.NvidiaDevicePluginManifest)
		setUpFunctions = append(setUpFunctions, deployDaemonSet("nvidia-device-plugin-daemonset", "kube-system"))
	}

	if *efaEnabled {
		deploymentManifests = append(deploymentManifests, manifests.EfaDevicePluginManifest)
		setUpFunctions = append(setUpFunctions, deployDaemonSet("aws-efa-k8s-device-plugin-daemonset", "kube-system"))
	}
	log.Printf("Using template variables: REGION=%s, VERSION=%s, VARIANT=%s, INSTANCE_TYPE=%s, TEAM_IDENTIFIER=%s, TEST_NAME=nvidia",
		region, manifestVersion, *amiVariant, *nodeType, *teamIdentifier)
	setUpFunctions = append(setUpFunctions, checkNodeTypes)
	testenv.Setup(setUpFunctions...)

	testenv.Finish(
		func(ctx context.Context, config *envconf.Config) (context.Context, error) {
			log.Println("Deleting MPI operator, DCGM Exporter, CloudWatch Agent and device plugin manifests.")
			if *efaEnabled {
				err := fwext.DeleteManifests(cfg.Client().RESTConfig(), manifests.EfaDevicePluginManifest)
				if err != nil {
					return ctx, err
				}

			}
			slices.Reverse(deploymentManifests)
			err := fwext.DeleteManifests(config.Client().RESTConfig(), deploymentManifests...)
			if err != nil {
				return ctx, err
			}

			log.Println("Successfully deleted MPI operator, DCGM Exporter, CloudWatch Agent and device plugin manifests.")
			return ctx, nil
		},
	)

	os.Exit(testenv.Run(m))
}
