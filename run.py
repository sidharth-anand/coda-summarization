import os
import sys

def preprocess():
    print('Running preprocess')

    if os.system('python preprocess.py > preprocess.log') == 0:
        print("Finished")
    else:
        print("Failed")

    sys.exit()

def train_a2c(start_reinforce, end_epoch, critic_pretrain_epochs, data_type, gpus):
    run = 'python train-a2c.py ' \
            '-data dataset/processed/processed_all.train.pickle ' \
            '-start_reinforce %s ' \
            '-end_epoch %s ' \
            '-critic_pretrain_epochs %s ' \
            '> a2c-train_%s_%s_%s_%s_g%s.test.log' \
            % (start_reinforce, end_epoch, critic_pretrain_epochs, start_reinforce, end_epoch, critic_pretrain_epochs, data_type, gpus)
    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()

def test_a2c(data_type, has_attn, gpus):
    run = 'python a2c-train.py ' \
            '-data dataset/train/processed_all.train.pt ' \
            '-load_from dataset/result/model_rf_hybrid_1_29_reinforce.pt ' \
            '-embedding_w2v dataset/train/ ' \
            '-eval -save_dir . ' \
            '-data_type %s ' \
            '-has_attn %s ' \
            '-gpus %s ' \
            '> a2c-test_%s_%s_%s.log' \
            % (data_type, has_attn, gpus, data_type, has_attn, gpus)
    print(run)
    a = os.system(run)
    if a == 0:
        print("finished.")
    else:
        print("failed.")
        sys.exit()

if __name__ == '__main__':
    if sys.argv[1] == 'preprocess':
        preprocess()

    if sys.argv[1] == 'train':
        train_a2c(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

    if sys.argv[1] == 'test':
        test_a2c(sys.argv[2], sys.argv[3], sys.argv[4])