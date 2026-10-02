"""The only place where Isaac Sim picks a GPU. Scripts must start the app via make_app().

run.sh sets:
  CUDA_VISIBLE_DEVICES=<N>  -> CUDA (physics, torch) sees only our card as cuda:0
  GT_VK_UUID + GT_VK_FILTER=1 -> implicit Vulkan layer (server/vk_filter) shows the renderer
                            only our card, so GT_KIT_GPU=0. Kit resets VK_INSTANCE_LAYERS,
                            so the layer is implicit (manifest in $XDG_DATA_HOME).
"""

import os
import sys

ASSET_ROOT = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"


def _env_gpu():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    kit = os.environ.get("GT_KIT_GPU", "")
    if not cvd.isdigit() or not kit.isdigit():
        sys.exit("[isaac_app] CUDA_VISIBLE_DEVICES and GT_KIT_GPU must be single numbers, use server/run.sh")
    if os.environ.get("GT_VK_FILTER") != "1" or not os.environ.get("GT_VK_UUID"):
        sys.exit("[isaac_app] Vulkan GPU filter layer is not enabled, use server/run.sh")
    return int(kit)


def make_app(width=640, height=480, extra_args=()):
    kit_gpu = _env_gpu()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "width": width,
            "height": height,
            "multi_gpu": False,
            "active_gpu": kit_gpu,
            "physics_gpu": 0,
            "extra_args": [
                "--/renderer/multiGpu/enabled=false",
                "--/renderer/multiGpu/autoEnable=false",
                "--/renderer/multiGpu/maxGpuCount=1",
                f"--/renderer/activeGpu={kit_gpu}",
                "--/plugins/carb.tasking.plugin/threadCount=4",
                # as in isaaclab.python.headless.kit, otherwise ISAAC_NUCLEUS_DIR is None
                f"--/persistent/isaac/asset_root/default={ASSET_ROOT}",
                f"--/persistent/isaac/asset_root/cloud={ASSET_ROOT}",
                f"--/persistent/isaac/asset_root/nvidia={ASSET_ROOT}",
                *extra_args,
            ],
        }
    )

    import carb

    s = carb.settings.get_settings()
    got = s.get("/renderer/activeGpu")
    multi = s.get("/renderer/multiGpu/enabled")
    print(f"[isaac_app] renderer activeGpu={got} (want {kit_gpu}), multiGpu={multi}", flush=True)
    if got is None or int(got) != kit_gpu or multi:
        print("[isaac_app] renderer is not pinned to our GPU, exiting", flush=True)
        os._exit(3)

    # what AppLauncher does for --enable_cameras in headless mode (isaaclab app_launcher.py)
    s.set_bool("/isaaclab/cameras_enabled", True)
    s.set_bool("/isaaclab/render/offscreen", True)
    s.set_bool("/isaaclab/render/active_viewport", False)
    s.set_bool("/isaaclab/render/rtx_sensors", False)
    s.set_bool("/physics/fabricUpdateTransformations", True)
    return app
