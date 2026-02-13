import json
import os
import socket

cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)


def get_node_rank():
    node_rank = int(cluster_spec["index"])
    return node_rank


def get_nnodes():
    nnodes = len(cluster_spec[role])
    return nnodes


def get_node_addr_ports(node_rank=None):
    if node_rank is None:
        node_rank = get_node_rank()
    node = cluster_spec[role][node_rank]
    node_addr, node_ports = node.split(":")
    node_addr = socket.gethostbyname(node_addr)
    node_ports = node_ports.split(",")
    return node_addr, node_ports


if __name__ == "__main__":
    master_addr, master_ports = get_node_addr_ports(node_rank=0)
    nnodes = get_nnodes()
    node_rank = get_node_rank()
    import torch
    nproc_per_node = torch.cuda.device_count()
    print(f"{node_rank} {master_addr} {master_ports[0]} {nnodes} {nnodes * nproc_per_node}")
