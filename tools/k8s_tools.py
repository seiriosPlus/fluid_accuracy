#!/bin/env python
import os
import sys
import time
import socket
from kubernetes import client, config
NAMESPACE = os.getenv("NAMESPACE")
if os.getenv("KUBERNETES_SERVICE_HOST", None):
    config.load_incluster_config()
else:
    config.load_kube_config()
v1 = client.CoreV1Api()

def get_pod_status(item):
    phase = item.status.phase

    # check terminate time although phase is Running.
    if item.metadata.deletion_timestamp != None:
        return "Terminating"

    return phase


def containers_all_ready(label_selector):

    def container_statuses_ready(item):
        container_statuses = item.status.container_statuses
    
        for status in container_statuses:
            if not status.ready:
                return False
        return True


    api_response = v1.list_namespaced_pod(
        namespace=NAMESPACE, pretty=True, label_selector=label_selector)

    for item in api_response.items:
        if not container_statuses_ready(item):
            print "[DEBUG] has contain not ready"
            return False

    return True


def fetch_pods_info(label_selector, phase=None):
    api_response = v1.list_namespaced_pod(
        namespace=NAMESPACE, pretty=True, label_selector=label_selector)
    pod_list = []
    for item in api_response.items:
        if phase is not None and get_pod_status(item) != phase:
            continue

        pod_list.append((item.status.phase, item.status.pod_ip))
    return pod_list


def wait_pods_running(label_selector, desired):
    print "label selector: %s, desired: %s" % (label_selector, desired)
    while True:
        count = count_pods_by_phase(label_selector, 'Running')
        # NOTE: pods may be scaled.
        if count >= int(desired):
            break
        print 'current cnt: %d sleep for 5 seconds...' % count
        time.sleep(5)

def wait_containers_ready(label_selector):
    print "label selector: %s, wait all containers ready" % (label_selector)
    while True:
        if containers_all_ready(label_selector):
            break
        print 'not all containers ready, sleep for 5 seconds...' 
        time.sleep(5)


def count_pods_by_phase(label_selector, phase):
    pod_list = fetch_pods_info(label_selector, phase)
    return len(pod_list)


def fetch_ips_list(label_selector, phase=None):
    pod_list = fetch_pods_info(label_selector, phase)
    ips = [item[1] for item in pod_list]
    ips.sort()
    return ips


def fetch_ips_string(label_selector, phase=None):
    ips = fetch_ips_list(label_selector, phase)
    return ",".join(ips)


def fetch_endpoints_string(label_selector, port, phase=None):
    ips = fetch_ips_list(label_selector, phase)
    ips = ["{0}:{1}".format(ip, port) for ip in ips]
    return ",".join(ips)


def fetch_pod_id(label_selector, phase=None):
    ips = fetch_ips_list(label_selector, phase=phase)

    local_ip = socket.gethostbyname(socket.gethostname())
    for i in xrange(len(ips)):
        if ips[i] == local_ip:
            return i

    return None


def fetch_ips(label_selector):
    return fetch_ips_string(label_selector, phase="Running")


def fetch_endpoints(label_selector, port):
    return fetch_endpoints_string(label_selector, port=port, phase="Running")


def fetch_id(label_selector):
    return fetch_pod_id(label_selector, phase="Running")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "fetch_ips":
        print fetch_ips(sys.argv[2])
    if command == "fetch_ips_string":
        print fetch_ips_string(sys.argv[2], sys.argv[3])
    elif command == "fetch_endpoints":
        print fetch_endpoints(sys.argv[2], sys.argv[3])
    elif command == "fetch_id":
        print fetch_id(sys.argv[2])
    elif command == "count_pods_by_phase":
        print count_pods_by_phase(sys.argv[2], sys.argv[3])
    elif command == "wait_pods_running":
        wait_pods_running(sys.argv[2], sys.argv[3])
    elif command == "wait_containers_ready":
        wait_containers_ready(sys.argv[2])
