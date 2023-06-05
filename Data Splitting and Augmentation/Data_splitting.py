import os, random
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide a Data folder')
    parser.add_argument('--path', default=' ', type=str, help='path to the data')
    parser.add_argument('--data', default='Total', type=str, help='Input folder name only, with out path ')
    args = parser.parse_args()


#Creating Train and Test Folder

list = ['TrainData','TestData','Train','Test_Lightness','Test_Rain']
for items in list: 
    os.mkdir(args.path+'/'+items)    

#Spliting total data into TrainData and TestData by 80:20 ration

for i in range(2):
    src_dir =args.path+'/'+args.data+'/'
    dst_dir =args.path+'/'+list[i]+'/'
    file_list = os.listdir(src_dir)
    print(len(file_list))
    if i==0:
        num=int(len(file_list)*8/10)
    else:
        num=len(file_list)
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    print(len(file_list))
    file_list = os.listdir(dst_dir)
    print(len(file_list))
shutil.rmtree(args.path+'/'+args.data+'/')


# Directory
list1 = ['a-5', 'b-4', 'c-3', 'd-2', 'e-1', 'f0', 'g1', 'h2', 'i3', 'j4', 'k5', 'l0','m10','n20','o30','p40','q50']
list2 = ['a-10', 'b-9', 'c-8', 'd-7', 'e-6', 'f-5', 'g-4', 'h-3', 'i-2', 'j-1', 'k0', 'l1', 'm2', 'n3', 'o4', 'p5', 'q6',
        'r7', 's8', 't9', 'u10']
list3=['a0','b10','c20','d30','e40','f50','g60','h70','i80','j90','k100']


#Creating directories in Train and test folders for weakly supervision and testing

parent_dir =args.path+'/'+list[2]+'/'
for items in list1:
    path = os.path.join(parent_dir, items)  
    os.mkdir(path)
parent_dir =args.path+'/'+list[3]+'/'
for items in list2:
    path = os.path.join(parent_dir, items)  
    os.mkdir(path)
parent_dir =args.path+'/'+list[4]+'/'
for items in list3:
    path = os.path.join(parent_dir, items)  
    os.mkdir(path)
    
    
#Moving training data into different folders of train data

src_dir =args.path+'/'+list[0]+'/'
N=len(os.listdir(src_dir))
for i in range(len(list1)):
    dst_dir =args.path+'/'+list[2]+'/'+list1[i]+'/'
    file_list = os.listdir(src_dir)
    if i<11:
        num=int(N/(2*11))
    else:
        num=int(N/(2*6))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
shutil.rmtree(args.path+'/'+list[0]+'/')

#Moving half of testing data into different folders of test lightness

src_dir =args.path+'/'+list[1]+'/'
N=int(len(os.listdir(src_dir))/2)
for i in range(len(list2)):
    dst_dir = args.path+'/'+list[3]+'/'+list2[i]+'/'
    file_list = os.listdir(src_dir)
    num=int(N/(21))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
    
#Moving remaining testing data into different folders of test rain

src_dir =args.path+'/'+list[1]+'/'
N=int(len(os.listdir(src_dir)))
for i in range(len(list3)):
    dst_dir = args.path+'/'+list[4]+'/'+list3[i]+'/'
    file_list = os.listdir(src_dir)
    num=int(N/(11))
    rem=int(len(file_list)-num)
    for i in range(num):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.move(src_dir + a, dst_dir + a)
    file_list = os.listdir(src_dir)
    file_list = os.listdir(dst_dir)
shutil.rmtree(args.path+'/'+list[1]+'/')
##AUROC folder
os.mkdir(args.path+'/'+'AUROC_Rain')
os.mkdir(args.path+'/'+'AUROC_Lightness')
os.mkdir(args.path+'/'+'AUROC_Rain'+'/'+'ID')
os.mkdir(args.path+'/'+'AUROC_Rain'+'/'+'OOD')
os.mkdir(args.path+'/'+'AUROC_Lightness'+'/'+'ID')
os.mkdir(args.path+'/'+'AUROC_Lightness'+'/'+'OOD')
L2 = sorted(os.listdir(path+'/'+'Test_Lightness'))
L3=sorted(os.listdir(path+'/'+'Test_Rain'))
for i in range(11):
    if i<=5:
        shutil.move(path+'/'+'Test_Rain'+'/'+L3[i], path+'/'+'AUROC_Rain'+'/'+'ID'+'/')
    else:
        shutil.move(path+'/'+'Test_Rain'+'/'+L3[i], path+'/'+'AUROC_Rain'+'/'+'OOD'+'/')
        
for i in range(21):
    if i<=4 or i>=16:
        shutil.move(path+'/'+'Test_Lightness'+'/'+L2[i], path+'/'+'AUROC_Lightness'+'/'+'OOD'+'/')
    else:
        shutil.move(path+'/'+'Test_Lightness'+'/'+L2[i], path+'/'+'AUROC_Lightness'+'/'+'ID'+'/') 
