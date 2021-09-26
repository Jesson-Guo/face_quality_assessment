import os


def make_training_dataset(dataset_path):
    f = open("training.txt", 'w')
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:  # 遍历文件夹
            if file_name.endswith('jpg'):
                f.write("/cache/data/" + file_name)
                file_name = file_name.replace('jpg', 'txt')
                info = open(dataset_path+file_name).readline().strip().split(" ")
                for i in info:
                    f.write("\t" + str(i))
                f.write("\n")
        break
    f.close()


make_training_dataset("D:\code\python\huawei-ascend\FQA\FaceQualityAssessment_code\dataset\AFLW2000\\")
