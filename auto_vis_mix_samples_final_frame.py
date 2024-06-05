import os
import sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())


actions = [
    'walking', 'sitting', 'eating', 'smoking',
    # 'posing', 'phoning'
]

for action in actions:
    vis = os.path.join(os.getcwd(), 'visualization', 'vis_gen.py')
    cmd = 'python ' + vis + ' --action ' + action
    print(cmd)
    os.system(cmd)