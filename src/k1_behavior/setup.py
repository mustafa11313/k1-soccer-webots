from setuptools import find_packages, setup

package_name = 'k1_behavior'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shk',
    maintainer_email='mustafaalmohammad123@gmail.com',
    description='TODO: Package description',
    license='AGPL-3.0-or-later',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ball_chaser = k1_behavior.ball_chaser:main',
            'striker = k1_behavior.striker:main',
        ],
    },
)
