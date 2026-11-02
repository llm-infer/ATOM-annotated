import os
import socket
import warnings

# 本文件：取本机对外通信用的IP地址
# 调用方：【分支】PD分离的mooncake / moriio connector、KV offload的Mooncake Store配置（见第9批），
#   用这个IP向对端公布自己的地址
# 与主干的关系：单机单节点的主干路径不用它（进程间走ipc://地址）


# 返回本机IP：优先环境变量ATOM_HOST_IP，否则用UDP"假连接"探测出口网卡的地址，都失败则返回"0.0.0.0"
def get_ip() -> str:
    # Check environment variable first
    # 【译】先检查环境变量
    # 注意：这里直接读os.environ，而不是通过atom.utils.envs
    host_ip = os.environ.get("ATOM_HOST_IP")
    # 只设置了旧的HOST_IP、没有ATOM_HOST_IP：发警告并忽略HOST_IP（Docker等软件常常设置它）
    if "HOST_IP" in os.environ and "ATOM_HOST_IP" not in os.environ:
        warnings.warn(
            "The environment variable HOST_IP is deprecated and ignored, as"
            " it is often used by Docker and other software to"
            " interact with the container's network stack. Please "
            "use ATOM_HOST_IP instead to set the IP address for ATOM processes"
            " to communicate with each other."
        )
    if host_ip:
        return host_ip

    # Try IPv4 by connecting to Google's DNS
    # 【译】尝试通过连接Google的DNS获取IPv4地址
    # UDP的connect()不发送任何数据包，只让内核选好路由和出口网卡；getsockname()拿到的就是那块网卡的本地IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 【译】（行尾注释）不需要真的能连通
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    # Try IPv6 by connecting to Google's IPv6 DNS
    # 【译】尝试通过连接Google的IPv6 DNS获取IPv6地址
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            # 【译】（行尾注释）不需要真的能连通
            s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
            return s.getsockname()[0]
    except Exception:
        pass

    # Fallback to 0.0.0.0 if all methods fail
    # 【译】所有方法都失败时，默认回退到0.0.0.0
    # 解释：下面警告里说"可以用ATOM_HOST_IP或HOST_IP设置"，但上面的代码其实会忽略HOST_IP
    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " ATOM_HOST_IP or HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"
