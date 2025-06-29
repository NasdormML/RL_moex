from setuptools import find_packages, setup

setup(
    name="moex_rl_agent",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "gym", "requests", "stable-baselines3"],
    author="Your Name",
    description="RL‑агент для торговли на MOEX",
    url="https://github.com/yourusername/MoexRLAgent",
)
