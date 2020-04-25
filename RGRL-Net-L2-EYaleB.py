import tensorflow as tf
import scipy.io as sio
import os

from AEnet import ConvAE
from AEutils import  *

# SELECT GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

 
        
def test_face(Img, Label, CAE, num_class,lr2=5e-4):
    # d = 10
    # alpha = 3.5
    # ro = 0.1# max(0.4 - (num_class-1)/10 * 0.1, 0.1)
    d = 4
    alpha = 3.5
    ro = 0.12
    acc_= []
    for i in range(0,39-num_class): 
        face_10_subjs = np.array(Img[64*i:64*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[64*i:64*(i+num_class)]) 
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs)    
        
            
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model
        
        max_step =  3000#3050 + num_class*25# 100+num_class*20
        display_step = 10000
        # fine-tune network
        epoch = 0
        COLD = None
        while epoch < max_step:
            epoch = epoch + 1
            cost, Coef = CAE.partial_fit(face_10_subjs, lr2, mode = 'fine')  #
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size))   )
                print(cost)
                # Coef = thrC(Coef,alpha)
                for posti in range(1):
                    display(Coef, label_10_subjs, d, alpha,ro)

            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(Coef - COLD, ord='fro')
                r = normcd/normc

                if r < 1.0e-6 and lastr < 1.0e-6:
                    print("early stop")
                    print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0] / float(batch_size)))
                    print(cost)
                    for posti in range(5):
                        display(Coef, label_10_subjs, d, alpha, ro)
                    break
                lastr = r
            COLD = Coef

        for posti in range(1):
            drawC(Coef)
            acc_x,_ ,_= display(Coef, label_10_subjs, d, alpha,ro)
            acc_.append(acc_x)
        acc_.append(acc_x)
        for sd in [5,6,7,9]:
            for sa in [3.5,4]:
                for sr in [0.16,0.14,0.12]:
                    print(sd, sa, sr)
                    display(Coef, label_10_subjs, sd, sa, sr)
    acc_ = np.array(acc_)
    mm = np.max(acc_)

    print("%d subjects:" % num_class)    
    print("Max: %.4f%%" % ((1-mm)*100))
    print(acc_) 
    
    return (1-mm)
    
if __name__ == '__main__':
    
    # load face images and labels
    data = sio.loadmat('./Data/YaleBCrop025.mat')
    img = data['Y']
    I = []
    Label = []
    for i in range(img.shape[2]):
        for j in range(img.shape[1]):
            temp = np.reshape(img[:,j,i],[42,48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label[:])
    Img = np.transpose(I,[0,2,1])
    Img = np.expand_dims(Img[:],3)
    # Img = np.divide(Img, 255.)

    model_path = './pretrain-model-EYaleB/model-102030-48x42-yaleb.ckpt'
    restore_path = './pretrain-model-EYaleB/model-102030-48x42-yaleb.ckpt'
    logs_path = './logs'

    # face image clustering
    n_input = [48,42]
    kernel_size = [5,3,3]
    n_hidden = [10,20,30]
    
    all_subjects = [38]#10, 15, 20, 25, 30, 35, 38]

    reg1 = 1e-4
    reg02 = 1.0  # 1.0 * 10 ** (num_class / 10.0 - 3.0)
    reg03 = 10
    mm = 0
    mreg=[0,0,0,0]


    results = []
    # for reg2 in [1e-5,1e-4,1e-3,1e-2]:
    #     for reg3 in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]:
    #         for reg4 in [0.01,0.1,1,10]:
    for reg2 in [1,10,100]:
        for reg3 in [1]:
            for reg4 in [100,1000]:
                for lr2 in [1e-4]:
                    try:
                        print("reg:", reg2, reg3,reg4, lr2)
                        avg = []
                        med = []
                        iter_loop = 0
                        while iter_loop < len(all_subjects):
                            num_class = all_subjects[iter_loop]
                            batch_size = num_class * 64

                            tf.reset_default_graph()
                            CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, re_constant3=reg3, re_constant4=reg4,\
                                         kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)

                            avg_i = test_face(Img, Label, CAE, num_class,lr2)
                            avg.append(avg_i)
                            iter_loop = iter_loop + 1

                        iter_loop = 0
                        results.append(1-avg[0])
                        if 1 - avg[0] > mm:
                            mreg= [reg2,reg3,reg4,lr2]
                            mm = 1 - avg[0]
                        print("max:", mreg, mm)
                    except:
                        print("error in ", reg2, reg3, lr2)
                    finally:
                        try:
                            CAE.sess.close()
                        except:
                            ''
    print(results)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
