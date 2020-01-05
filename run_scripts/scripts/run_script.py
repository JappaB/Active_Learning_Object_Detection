import os
import subprocess
import datetime



def main():
    run_scripts = [
        'boat_image_budget_pre-nms-avg_42_200.job',
        # '6class_image_budget_pre-nms-avg_42_200.job',
    ]


    curr_dir = os.getcwd()


    for script in run_scripts:
        print('Starting script: ', script)
        print('Time start: ', datetime.datetime.now())
        subprocess.call(['bash', script])

        print('Time stop: ', datetime.datetime.now())
        print('finished script :)!')
        print('________________________\n\n\n\n\n\n\n')

if __name__ == '__main__':
    main()
