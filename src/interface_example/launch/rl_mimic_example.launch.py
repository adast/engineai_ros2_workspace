from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('interface_example')

    # Get product and motion from environment variables or use defaults
    product = os.environ.get('PRODUCT', 'pm01')
    motion = os.environ.get('MOTION', 'pm01_flat')

    config_dir = os.path.join(package_dir, 'config', product, 'rl_mimic', motion)
    print(config_dir)

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    ENGINEAI_ROBOTICS_THIRD_PARTY = "/opt/engineai_robotics_third_party"
    os.environ['LD_LIBRARY_PATH'] = os.path.join(
        ENGINEAI_ROBOTICS_THIRD_PARTY, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    node = Node(
        package='interface_example',
        executable='rl_mimic_example',
        name='rl_mimic_example',
        arguments=[config_dir],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([node])
