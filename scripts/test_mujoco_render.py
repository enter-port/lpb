"""
Minimal script to verify MuJoCo + robosuite EGL offscreen rendering works.
Usage: python test_mujoco_render.py
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np


def test_robosuite_render():
    """Test robosuite environment creation + offscreen rendering."""
    import robosuite as suite

    print("=" * 50)
    print("[1/4] Creating robosuite environment (EGL offscreen)...")
    env = suite.make(
        env_name="Lift",
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
    )
    print("      Environment created OK")

    print("[2/4] Resetting environment...")
    obs = env.reset()
    print(f"      Reset OK, observation keys: {list(obs.keys())}")

    print("[3/4] Stepping environment + rendering...")
    for i in range(5):
        action = np.random.randn(env.action_dim)
        obs, reward, done, info = env.step(action)

    img = obs.get("agentview_image", None)
    if img is not None:
        print(f"      Render OK, image shape: {img.shape}, dtype: {img.dtype}")
    else:
        print("      WARNING: no agentview_image in obs, trying env.render()...")
        img = env.sim.render(128, 128, camera_name="agentview")
        print(f"      sim.render OK, image shape: {img.shape}")

    print("[4/4] Closing environment...")
    env.close()
    print("      Closed OK")

    print("=" * 50)
    print("ALL TESTS PASSED - MuJoCo rendering is working correctly.")


if __name__ == "__main__":
    test_robosuite_render()
