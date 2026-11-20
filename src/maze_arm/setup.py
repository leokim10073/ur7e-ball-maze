from setuptools import find_packages, setup

package_name = "maze_arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Chelsea Yan",
    maintainer_email="chelseayan@berkeley.edu",
    description="Arm control for the maze: board forward kinematics and remote-pivot tilt control.",
    license="MIT",
    tests_require=["pytest"],
)
