import datetime
import os


def create_save_dir(default_path = './outputs'):
    # 获取当前日期
    today = datetime.date.today().strftime('%Y-%m-%d')
    time = datetime.datetime.now().strftime('%H-%M-%S')
    save_path = os.path.join(default_path, today, time)

    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

if __name__ == '__main__':
    save_path = create_save_dir()
    print(save_path)