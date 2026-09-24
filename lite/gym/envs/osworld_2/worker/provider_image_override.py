"""Configure the release's Docker runtime image, guest RAM, network, and cleanup."""

import json
import os
from pathlib import Path

USER_NETWORK = (
    "-netdev user,id=hostnet0,host=20.20.20.1,net=20.20.20.0/24,dhcpstart=20.20.20.21,"
    "hostfwd=tcp::5000-20.20.20.21:5000,hostfwd=tcp::9222-20.20.20.21:9222,"
    "hostfwd=tcp::8080-20.20.20.21:8080,hostfwd=tcp::8000-20.20.20.21:8000,"
    "hostfwd=tcp::3000-20.20.20.21:3000 -device virtio-net-pci,romfile=,netdev=hostnet0,id=net0"
)


def apply(path):
    release_dir = Path(os.environ.get("OSWORLD_RELEASE_DIR", "/opt/osworld-release"))
    manifest = json.loads((release_dir / "release.json").read_text())
    source = path.read_text()
    literal = json.dumps(manifest["provider_images"]["docker"]["ubuntu"]["runtime_image"])
    if source.count(literal) != 1:
        raise ValueError("Pinned Docker provider no longer matches the expected image call")
    if 'os.environ.get("OSWORLD_DOCKER_IMAGE"' not in source:
        source = source.replace(literal, 'os.environ.get("OSWORLD_DOCKER_IMAGE", ' + literal + ')')
    if 'os.environ.get("OSWORLD_VM_NETWORK")' not in source:
        marker = '        self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}  # Modify if needed'
        if source.count(marker) != 1:
            raise ValueError("Pinned provider environment initializer changed")
        source = source.replace(marker, marker + '\n        if os.environ.get("OSWORLD_VM_NETWORK") == "user":\n'
                                + '            self.environment.update(NETWORK="N", ARGUMENTS=' + repr(USER_NETWORK) + ')')
    if 'os.environ.get("OSWORLD_VM_RAM_SIZE"' not in source:
        marker = '"RAM_SIZE": "4G"'
        if source.count(marker) != 1:
            raise ValueError("Pinned provider RAM initializer changed")
        source = source.replace(marker, '"RAM_SIZE": os.environ.get("OSWORLD_VM_RAM_SIZE", "4G")')
    # Provider retry paths also remove containers. Remove their anonymous
    # /storage volumes so a failed setup retry cannot accumulate disk usage.
    if "self.container.remove()" in source:
        if source.count("self.container.remove()") != 2:
            raise ValueError("Pinned provider container cleanup changed")
        source = source.replace("self.container.remove()", "self.container.remove(v=True)")
    path.write_text(source)


if __name__ == "__main__":
    apply(Path("/opt/osworld/desktop_env/providers/docker/provider.py"))
