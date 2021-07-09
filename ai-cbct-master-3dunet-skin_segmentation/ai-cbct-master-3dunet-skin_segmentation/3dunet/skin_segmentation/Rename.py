import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.

for j in range(1, 12):
    #file_path = 'C:\sers\user\Desktop\MaskCheck\2021-04-02\TP\TP_Ryu Eun-yeol_1996M' + str(j)
    file_path = 'D:/CT/DI_LeeJuHyeon/'
    file_names = os.listdir(file_path)
    i = 1
    for name in file_names:
        src = os.path.join(file_path, name)
        #dst = 'DD_JinMiRi_1994F_%04d_MASK' %i + '.dcm'
        dst = 'DI_LeeJuHyeon_%03d' % i + '.dcm'
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        i += 1