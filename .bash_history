killall -9 python
gpustat
nohup bash scripts/bash/train_diffusion.sh &>ddddlog.out&
gpustat
killall -9 python
killall -9 python3
nohup bash scripts/bash/train_diffusion.sh &>ddddlog.out&
gpustatt
gpustat
igpustat
gpustat
conda activate ldm
killall -9 python
gpustat
nohup bash scripts/bash/train_diffusion.sh &>log.out&
gpustat
conda activate ldm
i
killall -9 python3
killall -9 python
nohup bash scripts/bash/train_diffusion.sh &>log2.out&
gpustat
conda activate ldm
nohup bash scripts/bash/train_diffusion.sh &>log2.out&
gpustat
conda activate ldm
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
ls
cd stable-diffusion-webui/
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
bash webui.sh 
CUDA_VISIBLE_DEVICES=5 bash webui.sh 
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
ls
cd models/ControlNet/
wget https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_normal.pth
ssh zhicai.11
cd stable-diffusion-webui/
bash webui.sh 
gpustat
ls
bash weguish
bash wegui.sh
bash webui.sh 
CUDA_VISIBLE_DEVICES=5 bash webui.sh 
gpustat
du -h
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
cd /data/zhicai/dataset/dancedata/
ls
cd web-ui/
ls
rm -rf man
python -m http.server
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
cd /data/zhicai/dataset/dancedata/edn/subject1/train/train_img/
ls
cd ..
ls
cd primitives/
ls
cd processed/
ls
pwd
cd /data/zhicai/dataset/dancedata/web-ui/
ls
rm -rf edn_subject1/
rm -rf man/
ls
cd edn_subject1/
ls
cd ..
rm -rf edn_subject1/
cd edn_subject1/
ls
cd ..
ls
rm -rf man
gpustat
bash webui.sh 
CUDA_VISIBLE_DEVICES=5 bash webui.sh 
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
cd /data/zhicai
ls
cd dataset/
ls
cd dancedata/
ls
cd web-ui/
ls
pwd
cd /data/hizcia
cd /data/zhicai/
cd dataset/
cd dancedata/
cd edn/
ls
cd subject1/
ls
cd train/
ls
cd primitives/
ls
cd processed/
ls
pwd
cd ../../
ls
cd ..
ls
cd subject1/
ls
cd ..
ls
cd web-ui/
ls
cd edn_subject1/
ls
cd ..
ls
cd ..
ls
cd web-ui/
ls
cd edn_sb1_noB/
ls
rm \{i\}.png 
ls
exit
conda activate ldm
wget https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip
mv ../data.zip ./
unzip data.zip 
ln -s /data/zhicai/dataset/dancedata/web-ui/edn_sb1_noB/ ./data/reg_man
ln -s /data/zhicai/dataset/dancedata/web-ui/man/ ./data/reg_man
ln -s /data/zhicai/ckpts/ldm/ckpts/sd-v1-4-full-ema.ckpt ./models/
killall -9 python3
killall -9 python
ln -s /home/zhicai/stable-diffusion-webui/outputs/txt2img-images/2023-03-07 ./data_reg/reg_cat
ln -s /home/zhicai/stable-diffusion-webui/outputs/txt2img-images/2023-03-07/ ./data_reg/reg_cat/
ln -s /home/zhicai/stable-diffusion-webui/outputs/txt2img-images/2023-03-07/. ./data_reg/reg_cat/
bash scripts/finetune_real.sh "cat" data/cat data_reg/reg_cat/2023-03-07  cat finetune_addtoken.yaml models/sd-v1-4-full-ema.ckpt
pip install clip-retrieval
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
killall -9 python
conda activate ldm
gpustat
pip install pytorch-lightning==1.5.0
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt "<new1> man is dancing" --delta_ckpt logs/2023-03-07T16-44-01_man-sdv4/checkpoints/epoch=000004.ckpt --ckpt models/sd-v1-4-full-ema.ckpt
cd ..
ls
cd poseVideo/
ls
git clone git@github.com:adobe-research/custom-diffusion.git
ls
ls /data/zhicai/ckpts/
ls /data/zhicai/ckpts/ldm
ln -s /data/zhicai/ckpts/ldm/ckpts/ ./model
cd ..
ls
cd stable-diffusion-webui/
bash webui.sh 
CUDA_VISIBLE_DEVICES=5 bash webui.sh 
conda activate ldm
ls -s ../plDiffDance/ldm/ ./ldm/
ls -s ../plDiffDance/ldm ./ldm
ln -s ../plDiffDance/ldm/ ./ldm
gpustat
ls
ls ../
ls ../../
ln -s ../../stable-diffusion-webui/models/Stable-diffusion/ ./models/
ln -s ../../stable-diffusion-webui/models/Stable-diffusion/ ./models
ln -s /data/zhicai/ckpts/ldm/ckpts/ ./model
cd /data/zhicai/ckpts/ldm
ls
cd ckpts/
ls
cd kl-f
cd ..
ssh zhicai.10
gpustat
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
conda activate ldm
mv sd15_openpose.pth models/
du -h
ls
cd ~
ls
xsync stable-diffusion-webui/ -remote zhicai.12 zhicai.10
conda activate ldm
gpustat
bash scripts/finetune_gen.sh "man" data/edn_sb1 gen_reg/samples_man man finetune_addtoken.yaml  models/sd-v1-4-full-ema.ckpt
_VISIBLE_DEVICES=1 python sample.py --prompt "<new1> man" --delta_ckpt logs/2023-03-09T06-58-57_man-sdv4/checkpoints/epoch=000006.ckpt --ckpt models/sd-v1-4-full-ema.ckpt
cd ~
ls
cd bin/
ls
cd ..
ls
cd bin/
ls
cd ..
which openpose
cd /data/zhicai
cd ckpts/
ls
cd ldm/
ls
cd ckpts/
ls
cd ~/stable-diffusion-webui/
ls
cd models/
ls
cd S
cd ControlNet/
ls
conda activate ldm
python
python src/preprocess_pose_data.py
gpustat
killall -9 python3
killall -9 python
gpustat
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors
import torch
pythoh
python
conda activate ldm
ssh-copy-id -i ~/.ssh/id_rsa_lds13.pub zhicai.10
vim ~/.ssh/config 
ssh zhicai.1-
ssh zhicai.10
ln -s ~/stable-diffusion-webui/models/ControlNet/ ./models/
pip install safetensors
gpustat
python src/cldm/compose_ckpt.py 
ln -s ~/stable-diffusion-webui/models/openpose/ ./models/
gpustat
graphviz
apt install graphviz
cd ~
wget  https://graphviz.gitlab.io/pub/graphviz/stable/SOURCES/graphviz.tar.gz
unzip graphviz.tar.gz 
tar -xvf graphviz.tar.gz 
ls
cd graphviz-2.40.1/
./configure  --prefix=/home/zhicai/
make && make install
s
ls
cd ..
ls
cd lib/
ls
cd ..
ls
cd bin/
ls
cd ..
ls
cd share/
ls
cd man/
ls
cd ..
ls
ls ..
cd Be
cd ..
cd Benchmark/
ls
cd ..
rm Benchmark/ 
rm -rf Benchmark/ 
ls
cd bin/
l
ls
cd ..
ls
rm graphviz.tar.gz 
dot --version
dot -V
pip install torchviz
gpustat
cd //data/zhicai/dataset/dancedata/edn/subject2/val
cd /data/zhicai/dataset/dancedata/edn/subject
cd /data/zhicai/dataset/dancedata/edn
ls
ssh zhicai.10
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
bash webui.sh 
gpustat
CUDA_VISIBLE_DEVICES=1 bash webui.sh 
conda activate ldm
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors
killall -9 python3
killall -9 python
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors
conda activate ldm
ls -hl models
ls -hl models/
ls -lh logs/2023-03-10T14-12-12_man-sdv4/checkpoints/last.ckpt
conda activate ldm
gpustat
python src/cldm/compose_ckpt.py
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/sd15_openpose.ckpt
gpustat
killall -9 python
gpustat
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/sd15_openpose.ckpt
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml models/sd15_openpose.ckpt
conda activate ldm
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors
gpustat
exit
conda activate ldm
bash scripts/finetune_gen.sh "man" data/edn_subject1 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/sd15_openpose.ckpt
gpustat
ssh zhicai.12
xsync ./ -remote zhicai.12
gpustat
cd data/edn_subject1/images
ls -l
cd ..
ls -l
xsync /data/zhicai/dataset/dancedata/web-ui/edn_sb1_noB/ -remote zhicai.12
xsync /data/zhicai/dataset/dancedata/web-ui/man/ -remote zhicai.12
cd bin/
ls
cd ..
cd xy
cd bin/
vim xsync 
ls
cd ..
ls
rm -rf graphviz-2.40.1/
ls
cd lib/
ls
cd ..
rm -rf lib/
ls
cd Retrieval/
ls
cd ..
rm -r Retrieval/
cd share/
ls
cd ..
rm -rf share/
ls
cd include/
ls
cd ..
rm -rf include/
ls
cd Clothes/
ls
cd ..
rm -r Clothes/
ls
cd bin/
ls
mv xsync ../
ls
cd ..
rm -rf bin/
mkdir bin
mv xsync bin/
l
ls
xsync bin/xsync -remote zhicai.12
conda activate ldm
onda activate ldm cd /home/zhicai/poseVideo/custom-diffusion ; /usr/bin/env /data/zhicai/miniconda3/envs/ldm/bin/python /home/zhicai/.vscode-server/extensions/ms-python.python-2023.4.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 58479 -- /home/zhicai/poseVideo/custom-diffusion/src/preprocess_pose_data.py 
cp /home/zhicai/poseVideo/plDiffDance/src/impersonator_plus/iPERCore/tools/human_mattors/ annotator/
cp -r /home/zhicai/poseVideo/plDiffDance/src/impersonator_plus/iPERCore/tools/human_mattors/ annotator/
cp -r /home/zhicai/poseVideo/plDiffDance/src/impersonator_plus/assets/configs/detection annotator/
conda activate ldm
gpustat
cd /data/zhicai/dataset/
ls
cd dancedata/
ls
cd edn/
ls
cd subject
cd subject2
ls
cd val/
ls
cd test_
cd test_img/
ls
cd ..
cd /data/zhicai/dataset/dancedata/edn/subject1/val/primitives/processed/
ls
conda activate ldm
conda activate ldm
gpustat
killall -9 python
gpustat
ps -ef | grep zhicai
gpusta
gpustat
gpustat cd /home/zhicai/poseVideo/custom-diffusion ; /usr/bin/env /data/zhicai/miniconda3/envs/ldm/bin/python /home/zhicai/.vscode-server/extensions/ms-python.python-2023.4.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 52399 -- /home/zhicai/poseVideo/custom-diffusion/src/preprocess_pose_data.py 
gpustat
killall -9 python3
killall -9 python
ps -ef |grep zhicai
gpustat
bash scripts/sample.sh
conda activate ldm
bash scripts/sample.sh
conda activate ldm
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
conda activate ldm
gpustat
cd /data/zhicai/ckpts/
ls
cd ..
ls
cd logs/
ls
cd diffdance/
s
ls
python -m http.server
conda activate ldm
ls
cd ..
mv custom-diffusion control-diffusion
git clone https://github.com/adobe-research/custom-diffusion
git clone 
git clone https://github.com/Zhicaiwww/custom-diffusion
ls
git clone https://github.com/Zhicaiwww/custom-diffusion
scp control-diffusion/data.zip custom-diffusion/
cd custom-diffusion/
python split_dog.py
pip install clean-fid
cd /data/zhicai/dataset/
ls
mkdir custom-diffusion
cd custom-diffusion/
ls
cd retrieve/
ls
cd ..
cd retrieve/
ln -s ./ /home/zhicai/poseVideo/custom-diffusion/retrieve/
ln -s ./ /home/zhicai/poseVideo/custom-diffusion/retrieve
pwd
bash scripts/sample.sh 
conda activate ldm
bash scripts/sample.sh 
mkdir datas
mv data.zip datas/
cd datas
unzip data.zip 
ssh zhicai.8
cd ..
gpustat
cd ..
cd custom-diffusion/
ls
bash scripts/finetune_real.sh "dog" data/dog retrieve/retrieve_dog_200  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
gpustat
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
CUDA_VISIBLE_DEVICES=0 python sample.py --from-file data/retrieve_dog_200/caption.txt --delta_ckpt logs/2023-03-26T12-52-38_dog-sdv4/checkpoints/epoch=000005.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 2 --n_iter 1
CUDA_VISIBLE_DEVICES=0 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-26T12-52-38_dog-sdv4/checkpoints/epoch=000005.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 30
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken2.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken2.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
iq
cd ..lsils
ls
ilslsilsls
ls
conda activate ldm
unzip data.zip 
cd ..
ls
cd ..
ls
ln -s stable-diffusion-webui/models/ poseVideo/custom-diffusion/
cd /data/zhicai
ls
cd dataset/
ls
cd dancedata/
ls
tar -xvf dogs50B-val.tar.gz 
ls
cd dogs50B-val/
pwd
ssh zhicai.12
CUDA_VISIBLE_DEVICES=7 python sample.py --from-file data/retrieve_dog_200/caption.txt --delta_ckpt logs/2023-03-26T12-00-08_dog-sdv4/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 2 --n_iter 1
conda activate ldm
CUDA_VISIBLE_DEVICES=7 python sample.py --from-file data/retrieve_dog_200/caption.txt --delta_ckpt logs/2023-03-26T12-00-08_dog-sdv4/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 2 --n_iter 1
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-26T12-00-08_dog-sdv4/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 30
CUDA_VISIBLE_DEVICES=0 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-26T12-00-08_dog-sdv4/checkpoints/epoch=000002.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 30
cd /data/zhicai/logs/diffdance
ls
python -m http.server
gpustat
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
gpustat
exit
gpustat
exit 
gpustat
exit
gpustat
exit
gpustat
exit
gpustat
exit
gpustat
cd poseVideo/
ls
cd ..
ls
cd stable-diffusion-webui/
ls
bash webui.sh 
killall -9 python
killall -9 python3
bash webui.sh 
gpustat
exit
gpustat
conda activate ldm
ssh zhicai.9
gpustat
conda activate ldm
jobs
sleep 30 &
jobs
ps aux | grep zhicai
ps aux | grep zhicaidd python
ps aux | grep zhicaidd pythod
ps
ps -eH --forest | less
ps -ef | less
ps -u zhicai
ps -t
sleep 30s &
ps -
ps -u zhicai
ps -e | grep zhicai
ps -ef | grep zhicai
ps -ef | grep 'zhicai\|python'
gpustat
ps -ef | grep 'zhicai\|python'
ps -ef | grep 'zhicai|python'
ps -ef | grep 'zhicai'
ps -ef | grep 'sample_script'
disown -h 2303829 2303831 2303832 2303833 2303834 2303835
disown -h 2303829
ps -ef | grep 'sample_script'
gpustat
cat ~/.ssh/config 
ssh zhicai.8
exit
gpustat
exit
gpustat
exit
gpustat
exit
gpustat
exit
gpustat
conda activate ldm
bash training_scripts/run_lorpt.sh 
conda activate ldm
gpustat
conda activate ldm
gpustat
bash training_scripts/run_lorpt.sh 
conda activate ldm
cat /proc/sys/fs/inotify/max_user_watches
conda activate ldm
bash training_scripts/run_lorpt.sh 
gpustat
bash training_scripts/run_lorpt.sh 
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
conda activate ldm
cd ..
git clone https://github.com/mit-han-lab/fastcomposer.git
cd ..
code
cd poseVideo/
code fastcomposer/
conda activate ldm
gpustat
nohup bash training_scripts/run_lorpt.sh &>>log.out& 
gpustat
conda activate ldm
conda activate ldm
gpustat
nohup bash training_scripts/run_lorpt.sh &>>log.out&
nohup bash training_scripts/run_lorpt.sh &>>log.out2&
gpustat
nohup bash training_scripts/run_lorpt.sh &>>log.out2&
conda activate ldm
conda activate ldm
bash training_scripts/run_lorpt.sh 
conda activate ldm
chsh -s $(which zsh)
zsh
gpustconda activate ldm
at
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
cat /proc/sys/fs/inotify/max_user_watches
wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh
sh install.sh
ls
mv install.sh ~/
cd ~
bash install.sh 
cat install.sh 
zsh version
apt install zsh
sudo apt install zsh
su zyang
su duanjr
sh install.sh 
ls
rm -rf Repos/
ls
ls -a
cd .oh-my-zsh/
ls
cd ..
zsh
sh install_zsh.sh 
cd /usr/bin/zsh
ls
cd /usr/bin
ls
cd zsh
cd ~
zsh
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
killall -9 python
nohup bash training_scripts/run_lorpt.sh &>log.out&
nohup bash training_scripts/run_lorpt.sh &>log.out&zsh
zsh
gpustat
~
cd ~
code .zshrc
code install_zsh.sh 
sh install_zsh.sh clean
rm .zsh
rf -rf .zsh
rm -rf .zsh
rm -rf .zshrc
sh install_zsh.sh
type zsh
sh install_zsh.sh
zsh
bash train_script.py 
python train_script.py 
conda activate ldm
python train_script.py 
killall -9 python3
killall -9 python
gpustat
nohup python -u train_script.py &>>log.out&
iconda activate ldm
gputat
gpustat
nohup python -u sample_script.py &>>log.out&
conda activate ldm
nohup python -u sample_script.py &>>log.out&
gpustat
nohup python -u sample_script.py &>>log.out&
nohup python -u sample_script.py &>>log2.out&
conda activate ldm
sh scripts/finetune_real.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_real.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token <new1> --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token <new1> --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debugdd
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token <new1>python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token <new1>
gpustat
conda activate ldm
/data/zhicai/miniconda3/envs/ldm/bin/python
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
gpustat
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 1,3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
gpustat
killall -9 python3
killall -9 python
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 1,3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
killall -9 python
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 1,3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
killall -9 python3
killall -9 python
best_cudas | python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 0,1 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token '<new1>' --name dog --no-test --reg_scale 1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
gpustat
python train_script.py
gpustat
python sample_script.py 
gpustat
conda activate ldm
gpustat
bash train_script.py 
python train_script.py 
gpustat
best_cudas | python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 0,1 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token<new1> --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token<new1> --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
python train.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 5,6 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> dog' --datapath data/dog --modifier_token <new1> --name dog --no-test --reg_scale 0.1 --norm_scale 1 --postfix debug --reg_prompt 'a dog' --new_prompt 'a <new1> dog'
conda activate ldm
bash webui.sh 
source /home/zhicai/stable-diffusion-webui/venv/bin/activate
gpustat
conda activate ldm
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gconda activate ldm
pustat
gpustat
python train_script.py 
conda activate ldm
python train_script.py 
python sample_script.py 
python train_script.py 
gpustat
python train_script.py 
gpustat
cd logs
rm -rf 2023-04-06T16-59-31_teddybearreg_0_scale0_a_teddybear_lasso/
find . -type d -name "*teddybear*" -exec rm -rf {} +
gpustat
conda activate ldm
nohup python train_script.py &>>dog.out&
gpustat
nohup python train_script.py &>>teddybear.out&
gpustat
nohup python train_script.py &>>teddybear.out&
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python sample_script.py 
python train_script.py 
gpustat
python sample_script.py 
conda activate ldm
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python sample_script.py 
killall -9 python3
killall -9 python
python sample_script.py 
gpusta t
gpustat
python train_script.py 
cd /
df -h
cdconda activate ldm
cd logs
ls
du -h
gpustat
killall -9 python
gpustat
conda activate ldm
sleep 30 &
pconda activate ldm
python evaluation/clip_eval.py
conda activate ldm
python evaluation/clip_eval.py
gpustat
cd /data/liox
ls
ssh zhicai.12
conda activate ldm
ssh zhicai.12
gpustat
python sample_script.py 
gpustat
killall -9 python
python sample_script.py 
killall -9 python
python sample_script.py 
killall -9 python
python sample_script.py 
gpustat
nohup python train_script.py &>>log.out&
killall -9 python
nohup python train_script.py &>>log.out&
gpustat
killall -9 python
nohup python train_script.py &>>log.out&
gpustat
nohup python train_script.py &>>log.out&
gpusat
gpustat
ssh zhicai.12
gpustat
nohup python train_script.py &>>log.out&
gpustat
ssh zhicai.12
conda activate ldm
gpustat
git status
git add .gitignore 
git status
git add .
git rm --cached tools/BLIP
git rm --cached tools/BLIP -f
git add .
git status
git commit -m 'emphasis_enable'
xsync ./src2/ -remote zhicai.12
conda activate ldm
python src2/custom_modules.py 
gpustat
python sample_script.py 
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt '(<new1>:2) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt '(<new1>:1.2) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '(<new1>:1.2) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
conda activate ldm
gpustat
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:4) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:1.5) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:3) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
gpustat
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:0) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt 'tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
conda activate ldm
xsync ./ -remote zhicai.12
xsync ./logs/logs_toTest/2023-04-06T16-54-16_dogreg_0_scale0_a_dog_lasso/ -remote zhicai.12
python download.py 
conda activate ldm
python download.py 
cd ~/.cache/
ls
cd huggingface/
ls
xsync ./ -remote zhicai.10
conda activate ldm
xsync logs/logs_toTest/2023-04-08T04-49-07_teddybearreg_0_scale0_a_teddybear_ridge_onlyK_noblip/checkpoints -remote zhicai.12
gpustat
ssh zhicai.12
conda activate ldm
vim ~/bin/xsync 
xsync logs/logs_toTest/ -remote zhicai.10
ssh zhicai.12
ssh zhicai.9
ssh zhicai.10
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:1.5) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6conda activate ldm
CUDA_VISIBLE_DEVICES=6 python sample.py --prompt '(<new1>:2.0) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
conda activate ldm
CUDA_VISIBLE_DEVICES=6 python sample.py --prompt '(<new1>:2.0) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=6 python sample.py --prompt '(<new1>:1.0) tortoise plushy working on the laptop' --delta_ckpt logs/2023-04-08T03-32-12_tortoise_plushyreg_0.1_scale0_a_tortoise_plushy_ridge_onlyK_noblip/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
gpustat
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:1.2) teddybear wearing headphones' --delta_ckpt logs/2023-04-07T20-53-38_teddybearreg_1_scale0_a_teddybear_ridge_onlyK/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6I
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:1.1) teddybear wearing headphones' --delta_ckpt logs/2023-04-07T20-53-38_teddybearreg_1_scale0_a_teddybear_ridge_onlyK/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '(<new1>:1.5) teddybear wearing headphones' --delta_ckpt logs/2023-04-07T20-53-38_teddybearreg_1_scale0_a_teddybear_ridge_onlyK/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 1 --scale 6
git add .
git commit .
gpustat
conda activate ldm
gpustat
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt 'a <new1> cat and a <new2> dog' --delta_ckpt logs/2023-04-17T16-55-13_cat_doga_cat_dog_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000015-step=000000199.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
conda activate ldm
python sample_script.py 
python train_script.py 
gpustat
python sample_script.py 
conda activate ldm
python sample_script.py 
gpustat
ssh zhicai.12
gpustat
conda activate ldm
gpustat
python sample_script.py 
python train_script.py 
python sample_script.py
gpustat
python sample_script.py
python train_script.py 
conda activate ldm
gpsutat
gpustat
python sample_script.py 
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt 'a cat' --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 2 --scale 6 --sampling_method dpm --ddim_steps 10
gpustat
python train_script.py 
python sample_script.py 
python train_script.py 
nohup python train_script.py &>log.out&
gpustat
bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  chair+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
conda activate ldm
bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  chair+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_joint.sh "wooden pot" data/wooden_pot real_reg/samples_wooden_pot "cat" data/cat real_reg/samples_cat wooden_pot+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
gpustat
python sample_script.py 
gpustat
gpustay
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gpustat
nohup bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  chair+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt &>>log.out&
python train_script.py 
conda activate ldm
conda activate ldm
killall -9 python
python train_script.py 
gpustat
killall -9 python
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python sample_script.py 
nohup python train_script.py &>log.out& 
gpustat
jobs
job
conda activate ldm
nohup python train_script.py &>log.out& 
gpustat
conda activate ldm
gpustat
killall -9 python
nohup python train_script.py &>log.out& 
conda activate ldm
cd /data/liox/custom-diffusion/
ls
cd data/
ls
cd ..
cd imgs/
ls
cd for_ood/
ls
python -m http.server
conda activate ldm
killall -9 python 3
killall -9 python 
gpustat
nohup python train_script.py &>log.ou&
conda activate ldm
killall -9 python
gpustat
nohup python train_script.py &>log.ou&
gpustat
nohup python train_script.py &>log.ou&
killall -9 python
nohup python train_script.py &>log.ou&
gpustat
nohup python train_script.py &>log.out&
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
killall -9 python
python train_script.py 
gpustat
python sample_script.py 
gpustat
python train_script.py 
lscpu
python train_script.py 
conda activate ldm
python train_script.py 
gpustat
gconda activate ldm
pustat
gpustat
htop
conda activate ldm
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
killall -9 python
killall -9 python3
python train_script.py 
killall -9 python
python src2/custom_modules.py 
python train_script.py 
nohup python train_script.py &>>log.out&
gpustat
gpsutat
gpustat
conda activate ldm
python sample_script.py 
cp ../lora-master/ ./tools/
cp -r ../lora-master/ ./tools/
conda activate ldm
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T18-35-05_tortoise_plushy_classBias_reg0.1-0.1_scale0-0_1e-5/checkpoints/epoch=000007-step=000000119.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
gpustat
python train_script.py 
nohup python train_script.py &>log2.out&
gpustat
killall -9 python
nohup python train_script.py &>log2.out&
nohup python train_script.py &>log1.out&
gpusutat
gpustat
python sample_script.py 
conda activate ldm
python sample_script.py 
python train_script.py 
conda activate ldm
python sample_script.py 
gpustat
python sample_script.py 
gpustat
nohup python sample_script.py &>log.out&
gpustat
nohup python sample_script.py &>log.out&
gpustat
conda activate ldm
gpustat
ps -grep zhicai
ps -ef | grep zhicai
ps -ef | grep 727932
gpusutat
gpustat
python sample_script.py 
gpusta
gpustat
python train_script.py 
gpustat
python sample_script.py 
gpustat
conda activate ldm
gpustat
xsync ./ -remote zhicai.12
gpustat
python train_script.py 
gpustat
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'cat' --delta_ckpt logs/logs_liox/2023-04-05T05-00-30_dog-sdv4/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 8 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'a <new1> cat and a <new2> dog' --delta_ckpt logs/2023-04-17T14-59-43_cat_dogreg_0.1-0.1_scale0-0_a_cat_dog/checkpoints/epoch=000015-step=000000199.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'a <new1> cat and a <new2> dog' --delta_ckpt logs/2023-04-17T14-59-43_cat_dogreg_0.1-0.1_scale0-0_a_cat_dog/checkpoints/epoch=000038-step=000000499.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'a <new2> dog' --delta_ckpt logs/2023-04-17T14-59-43_cat_dogreg_0.1-0.1_scale0-0_a_cat_dog/checkpoints/epoch=000038-step=000000499.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'a <new2> dog playing with a <new1> cat' --delta_ckpt logs/2023-04-17T14-59-43_cat_dogreg_0.1-0.1_scale0-0_a_cat_dog/checkpoints/epoch=000038-step=000000499.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=4 python sample.py --prompt 'a dog playing with a cat' --delta_ckpt logs/2023-04-17T14-59-43_cat_dogreg_0.1-0.1_scale0-0_a_cat_dog/checkpoints/epoch=000038-step=000000499.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt 'The <new1> cat is sitting inside a <new2> wooden pot and looking up' --delta_ckpt logs/2023-04-17T17-21-57_cat_chaira_cat_chair_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000039-step=000000399.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6 --sampling_method dpm --ddim_steps 10
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt '<new2> chair with the <new1> cat sitting on it near a beach' --delta_ckpt logs/2023-04-17T17-21-57_cat_chaira_cat_chair_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000039-step=000000399.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6 --sampling_method dpm --ddim_steps 10
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt '<new2> chair' --delta_ckpt logs/2023-04-17T17-21-57_cat_chaira_cat_chair_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000039-step=000000399.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 8 --n_samples 4 --n_iter 2 --scale 6 --sampling_method dpm --ddim_steps 10
gpustat
bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  wooden_pot+cat finetune_joint.yaml --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  wooden_pot+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
gpustat
nohup bash scripts/finetune_joint.sh "chair" data/chair real_reg/samples_chair "cat" data/cat real_reg/samples_cat  wooden_pot+cat finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt &>>log.out&
conda activate ldm
unzip lora-master.zip 
mv -r lora-master ../
mv  lora-master ../
rm lora-master.zip 
cd ..
gpustat
cd custom-diffusion/
python sample_script.py 
gpustat
python train_script.py 
conda activate ldm
gpustat
python train_script.py 
gpustat
python train_script.py 
killall -9 python
python train_script.py 
killall -9 python
python train_script.py 
killall -9 python
python train_script.py 
gpustat
xsync ./ -remote zhicai.12
python sample_script.py 
gpustat
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath /data/tortoise_plushy --modifier_token <new1> --name tortoise_plushy --no-test --reg_k_scale 0.1 --norm_k_scale 0 --reg_v_scale 0.1 --norm_v_scale 0 --postfix 'debug' --reg_prompt_file data_reg/tortoise_plushy_reg.txt  --new_prompt 'new_prompt' --repeat 50 --concept_classes '<new1> tortoise plushy'
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml 
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath /data/tortoise_plushy --modifier_token <new1>
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath /data/tortoise_plushy --modifier_token '<new1>'
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath data/tortoise_plushy --modifier_token '<new1>' --name tortoise_plushy --no-test --reg_k_scale 0.1 --norm_k_scale 0 --reg_v_scale 0.1 --norm_v_scale 0 --postfix 'debug' --reg_prompt_file data_reg/tortoise_plushy_reg.txt  --new_prompt 'new_prompt' --repeat 50 --concept_classes '<new1> tortoise plushy'
gpustat
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus '3' --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath data/tortoise_plushy --modifier_token '<new1>' --name tortoise_plushy --no-test --reg_k_scale 0.1 --norm_k_scale 0 --reg_v_scale 0.1 --norm_v_scale 0 --postfix 'debug' --reg_prompt_file data_reg/tortoise_plushy_reg.txt  --new_prompt 'new_prompt' --repeat 50 --concept_classes '<new1> tortoise plushy'I
gpustat
killall -9 python
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus '4' --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath data/tortoise_plushy --modifier_token '<new1>' --name tortoise_plushy --no-test --reg_k_scale 0.1 --norm_k_scale 0 --reg_v_scale 0.1 --norm_v_scale 0 --postfix 'debug' --reg_prompt_file data_reg/tortoise_plushy_reg.txt  --new_prompt 'new_prompt' --repeat 50 --concept_classes '<new1> tortoise plushy'
python train2.py --base configs/custom-diffusion/finetune_addtoken2.yaml -t --gpus 3,4 --resume-from-checkpoint-custom Stable-diffusion/sd-v1-4-full-ema.ckpt --caption '<new1> tortoise plushy' --datapath data/tortoise_plushy --modifier_token '<new1>' --name tortoise_plushy --no-test --reg_k_scale 0.1 --norm_k_scale 0 --reg_v_scale 0.1 --norm_v_scale 0 --postfix 'debug' --reg_prompt_file data_reg/tortoise_plushy_reg.txt  --new_prompt 'new_prompt' --repeat 50 --concept_classes '<new1> tortoise plushy'
gpustat
conda activate ldm
ps -ef | grep zhicai
pstree
pstree -p
pstree -p 727301
gpustat
python sample_script.py 
gpusat
gpustat
xsync ./ -remote zhicai.12
gpustat
bash scripts/finetune_joint.sh "cat" data/cat real_reg/samples_cat "chair" data/chair real_reg/samples_chair  cat+chair finetune_joint.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python sample_script.py 
gpustat
python sample_script.py 
cd /data/liox/Dreambooth-Stable-Diffusion/imgs/for_overfit
python sample_script.py 
conda activate ldm
python sample_script.py 
gpustat
python sample_script.py 
killall -9 python
python sample_script.py 
cd ..
git clone git@github.com:cloneofsimo/lora.git
ls
git clone https://github.com/cloneofsimo/lora.git
gpsutat
gpustat
killall -9 python
gpustat
cd custom-diffusion/
python train_script.py 
python sample_script.py 
gpustat
python sample_script.py 
gpustat
jobs
fg 1
gpustat
python sample_script.py 
gpustat
python train_script.py 
python sample_script.py 
conda activate ldm
bash train_lora.sh 
conda activate ldm
bash train_lora.sh 
bash train_lora.sh  cd /home/zhicai/poseVideo/lora-master ; /usr/bin/env /data/zhicai/miniconda3/envs/ldm/bin/python /home/zhicai/.vscode-server/extensions/ms-python.python-2023.6.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 37129 -- training_scripts/train_lora_w_ti.py --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 --instance_data_dir=dataset/data/dog --output_dir=./output_dog --class_data_dir=dataset/real_reg/samples_dog/dog --class_prompt_or_file=dataset/real_reg/samples_dog/caption.txt --resolution=512 --train_batch_size=1 --learning_rate=1e-5 --learning_rate_text=1e-5 --learning_rate_ti=5e-4 --color_jitter --lr_scheduler=constant --lr_warmup_steps=100 --max_train_steps=2000 --placeholder_token=\<krk\> --learnable_property=object --initializer_token=dog --save_steps=500 --resize=True --center_crop --lora_rank=1 --gradient_accumulation_steps=4 --output_format=safe --unfreeze_lora_step=1000 
conda activate ldm
conda activate ldm
killall -9 python
bash training_scripts/run_lorpt.sh 
conda activate ldm
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat

python train_script.py 
gpustat
python train_script.py 
gpustat
ssh zhicai.12
gpustat
python train_script.py 
gpustat
python sample_script.py 
gpustat
python train_script.py 
gpustat
python train_script.py 
gpustat
python sample_script.py 
python train_script.pydd
python sample_script.py 
python train_script.py 
nohup python train_script.py &>log.out&
gpustat
nohup python sample_script.py &>log.out&
gpustat
python sample_script.py 
killall -9 python
cd ..
ls
cd control-diffusion/
ls
cd configs/
ls
cd cldm/
ls
cd ..
git clone 
git clone git@github.com:Stability-AI/stablediffusion.git
git clone https://github.com/Stability-AI/stablediffusion.git
cd custom-diffusion/
ls
mv stablediffusion-main.zip ../
cd ..
unzip stablediffusion-main.zip 
cp custom-diffusion/src stablediffusion-main/src/
cp custom-diffusion/src stablediffusion-main/src
cp -r custom-diffusion/src stablediffusion-main/src/
cp -r custom-diffusion/src2 stablediffusion-main/src2
cp -r custom-diffusion/train.py stablediffusion-main/

gpustat
cd custom-diffusion/
python train_script.py 
killall -9 python
nohup python train_script.py &>log.out&
python sample_script.py 
gpustat
python train_script.py 
git status
gpustat
cd ~ 
git clone https://github.com/Zhicaiwww/custom-diffusion
git clone git@github.com:Zhicaiwww/custom-diffusion.git
gpustat
git clone git@github.com:Zhicaiwww/custom-diffusion.git
ssh zhicai.12
git status
cd poseVideo/
cd custom-diffusion/
git status
python src/utils/util.py 
conda activate ldm
python src/utils/util.py 
python -m  src/utils/util.py 
python src/utils/util.py 
conda activate ldm
bash train
bash train_lora.sh 
gpustat
conda activate ldm
bash training_scripts/run_lorpt.sh 
conda activate ldm
conda activate ldm
cd training_scripts/
gpustat
killall -9 python
conda activate ldm
gpustat
bash train_lora.sh 
python setup.py 
python setup.py install
python setup.py 
bash train_lora.sh 
pip install transformers==4.25.1
pip install mediapipe
pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install transformers==4.25.1
bash train_lora.sh 
conda activate ldm
python sample_script.py 
killall -9 python
python sample_script.py 
python train_script.py 
python sample_script.py 
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt 'picture of a turtle' --delta_ckpt logs/logs_liox/2023-04-05T14-20-45_tortoise_plushy-sdv4/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
gpustat
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt 'picture of a turtle' --delta_ckpt logs/2023-04-22T17-07-46_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000006-step=000000099.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
gpustat
python train_script.py 
killall -9 python
python train_script.py 
killall -9 python
python train_script.py 
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T18-57-58_tortoise_plushy_lr5e-5classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000005-step=000000089.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T18-57-58_tortoise_plushy_lr5e-5classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000005-step=000000089.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/custom-diffusion/2023-04-22T19-01-39_tortoise_plushy_lr5e-6classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000005-step=000000089.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=7 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T19-01-39_tortoise_plushy_lr5e-6classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000007-step=000000119.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 1 --scale 6 --sampling_method ddim --ddim_steps 50
python train_script.py 
killall -9 python
cd cconda activate ldm
cd ../custom-diffusion/
cd dataset/
ls
cd ..
ln -s dataset/ ../lora-master/dataset/
cd dataset/
ls
ln -s dataset/ ../lora-master/dataset
ln -s dataset/ ../lora-master/dataset/
ln -s dataset ../lora-master/dataset/
ln -s dataset/ ../lora-master/dataset/
cd ..
ln -s dataset/ ../lora-master/dataset/
ln -s dataset/ ../lora-master/dataset
ln -s dataset/ ../lora-master/dataset/
ln -s dataset/ ../lora-master/dataset
cd dataset/
ls
gpustat
python
conda activate lmd
conda activate ldm
python 
gpustat
conda activate ldm
gpustat
bash training_scripts/run_lorpt.sh 
gpustat
killall -9 python
conda activate ldm
gpustat
conda activate ldm
conda activate ldm
nohup bash training_scripts/run_lorpt.sh &>>log.out&
gpustat
bash train_lora.sh 
bash training_scripts/run_lorpt.sh 
killall -9 python
nohup bash training_scripts/run_lorpt.sh &>log.out&
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
gpustat
conda activate ldm
bash training_scripts/run_lorpt.sh 
nohup bash training_scripts/run_lorpt.sh &>log.out&
ls -a output_dog/logs/dreambooth
ls -l output_dog/logs/dreambooth
ls - loutput_dog_PPD+caption
ls -l output_dog_PPD+caption
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
gpustat
nohup bash training_scripts/run_lorpt.sh &>log2.out&
nohup bash training_scripts/run_lorpt.sh &>log3.out&
gpustat
bash training_scripts/run_lorpt.sh 
gpustat
bash training_scripts/run_lorpt.sh 
conda activate ldm
gpustat
ssh zhicai.12
gpustat
bash train_lora.sh 
conda activate ldm
gpustat
ls
python sample_script.py 
gpustat
conda activate ldm
CUDA_VISIBLE_DEVICES=5 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/logs_liox/2023-04-05T05-04-21_cat-sdv4/checkpoints/delta_epoch=last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
gpustat
cd ..
git clone 
git clone https://github.com/mkshing/svdiff-pytorch
cd custom-diffusion/
python sample_script.py 
gpustat
python train_script.py 
_VISIBLE_DEVICES=3 python sample.py --prompt 'picture of a turtule' --delta_ckpt logs/2023-04-22T17-40-38_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'picture of a turtule' --delta_ckpt logs/2023-04-22T17-40-38_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'picture of a turtule' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000001-step=000000029.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '(<new1>:0.5) tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000001-step=000000029.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '(<new1>:0.8) tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000001-step=000000029.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '(<new1>:0.9) tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000001-step=000000029.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '(<new1>:0.8) tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt 'a <new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
killall -9 python
nohup python train_script.py &>log.out&
gpustat
nohup python train_script.py &>log1.out&
cd ~/.cache/
ls
cd huggingface/
ls
cd hub/
ls
cd !
cd -
cd ~/poseVideo/
cd lora-master/
ls
cd /home/zhicai/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/
ls
cd /home/zhicai/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345
cd aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/
ls
cd ..
ls
cd 3
cd 39593d5650112b4cc580433f6b0435385882d819/
ls
cd ..
ls
gpusta
gpustat
conda activate ldm
python sample_script.py 
gpustat
nvidia-smi
python sample_script.py 
gpustat
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-07-46_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000006-step=000000099.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-07-46_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000006-step=000000099.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-07-46_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000006-step=000000099.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 4 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in times square' --delta_ckpt logs/2023-04-22T17-07-46_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/epoch=000006-step=000000099.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-40-38_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 4 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt 'tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
CUDA_VISIBLE_DEVICES=1 python sample.py --prompt '<new1> tortoise plushy in grand canyon' --delta_ckpt logs/2023-04-22T17-53-27_tortoise_plushy_classBias_reg0.1-0.1_scale0-0/checkpoints/last.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_rows 4 --n_samples 8 --n_iter 2 --scale 6 --sampling_method ddim --ddim_steps 50
python train_script.py 
gpustat
python train_script.py 
killall -9 python
python train_script.py 
gpustat
conda activate ldm
gpustat
bash training_scripts/run_lorpt.sh 
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
nohup bash training_scripts/run_lorpt.sh &>log2.out&
nohup bash training_scripts/run_lorpt.sh &>log3.out&
gpustat
nohup bash training_scripts/run_lorpt.sh &>log3.out&
nohup bash training_scripts/run_lorpt.sh &>log2.out&
codna activate ldm
conda activate ld
conda activate ldm
ls
bash training_scripts/run_lorpt.sh 
gpustat
bash training_scripts/run_lorpt.sh 
gpustat
gconda activate ldm
gpustat
conda activate ldm
gpustat
nohup bash train_lora.sh &>log.out&
gpustat
ls -l exps/output_dog_0.001/final_lora.safetensors
ls -l exps/output_dog_0.01/step_300.safetensors
nohup bash train_lora.sh &>log.out&
git status
git init
git status
git add .gitignore 
git status
git add .
git rm -rf example_loras
git add .
git commit -m "inited "
pwd
ls
rm -rf dataset 
gpustat
conda activate ldm
gpustat
bash training_scripts/run_lora.sh 
bash training_scripts/run_lorpt.sh 
gpustat
killall -9 python
bash training_scripts/run_lorpt.sh 
gpustat
bash training_scripts/run_lorpt.sh 
conda activate ldm
gpustat
bash training_scripts/run_lorpt.sh 
nohup bash training_scripts/run_lorpt.sh &>log.out&
gpustat
bash training_scripts/run_lorpt.sh 
gpustat
conda activate ldm
bash training_scripts/run_lorpt.sh 
conda activate ldm
bash training_scripts/run_lorpt.sh 
gpustat
bash training_scripts/run_lorpt.sh 
nohuo bash training_scripts/run_lorpt.sh &>>log.out&
bash training_scripts/run_lorpt.sh 
conda activate ldm
bash training_scripts/run_lorpt.sh 
pip list
pip uninstall lora_diffusion
bash training_scripts/run_lorpt.sh 

bash training_scripts/run_lorpt.sh 
gpustat
nohup bash training_scripts/run_lorpt.sh &>log.out&
killall -9 python
nohup bash training_scripts/run_lorpt.sh &>log.out&
killall -9 python3
killall -9 python
conda activate ldm
python src/preprocess_pose_data.py
conda activate ldm
conda activate ldm
gpustat
bash scripts/finetune_gen.sh
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
[ -d "${ARRAY[5]}" ]
echo [ -d "${ARRAY[5]}" ]
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
conda activate ldm
check_gpu 2 20000 | python src/preprocess_pose_data.py
conda activate ldm
ssh zhicai.12
cd /data/zhicai/dataset/dancedata/edn/
ls
cd subject
cd subject1
ls
cd val/
ls
cd primitives/
ls
cd processed/
ls
cd 2Dpose/
pwd
gpustat
conda activate ldm
gpustat
rm -rf /data/zhicai/dataset/dancedata/edn/subject2/train/edn_subject2_train
gpustat
gpustat
xsync ./ -remote zhicai.12
ls
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic1 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
conda activate ldm
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic1 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
gpustat
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic1 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
cd /data/zhicai/dataset/
cd dancedata/
cd processed/
ls
cd edn/
ls
python src/preprocess_pose_data.py
conda activate ldm
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/custom-diffusion/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors "photo of a woman, best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, modern dressing"
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/custom-diffusion/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors 'photo of a woman, best quality, (photorealistic:1.2), a complete torso,colored picture, simple background, modern dressing'
python src/preprocess_pose_data.py 
conda activate ldm
bash scripts/sample.sh 
conda activate ldm
conda activate ldm
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train50 gen_reg/edn_subject2_train50 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train50 gen_reg/edn_subject2_train50 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pthbash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
killall -9 python3
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
killall -9 python3
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
check_gpu 2 20000 | python src/preprocess_pose_data.py
conda activate ldm
gpustat
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic2 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
gpustat
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic2 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
python src/preprocess_pose_data.py
src/preprocess_pose_data.py
python src/preprocess_pose_data.py
gpustat
codconda activate ldm
code ~/bin/check_gpu.sh 
memory=3000
free_gpus=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F "," '$2 ~ /MiB/ && $2 + 0 > $memory {print $1}' | tr "\n" "," | sed 's/,$//')
echo $free_gpus 
cd ~/bin/
ls
chmod +x check_gpu.sh
check_gpu 1 3000
mv check_gpu.sh check_gpu
check_gpu 1 3000
code check_gpu 
check_gpu 1 3000
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train50 gen_reg/edn_subject2_train50 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
gpustat
cd /data/zhicai/dataset/dancedata/edn
cd subject1
ls
cd train/
ls
rm -rf meta.npy 
cd /data/zhicai
lsc
ls
cd dataset/
ls
cd dancedata/
ls
cd processed/
ls
pwd
gpustat
cd ~
ls
cd poseVideo/
ls
cd ~
ls
cd poseVideo/
ls
cd custom-diffusion/
ls
du -h
mv data/edn_subject2_val /data/zhicai/dataset/dancedata/processed/
ln -s /data/zhicai/dataset/dancedata/processed/edn_subject2_val/ data/
du -h
ls models/openpose/
ls -l models/openpose/
ls -a models/openpose/
xsync ./ -remote zhicai.12
readlink models/openpose
xsync /home/zhicai/stable-diffusion-webui/models/openpose/ -remote zhicai.12
code ~/bin/xsync 
xsync ./ -remote zhicai.12
readlink data/edn_subject1_train20
mv /data/zhicai/dataset/dancedata/edn/subject1/train/edn_subject1_train20 /data/zhicai/dataset/dancedata/processed/
cd /data/zhicai/dataset/dancedata/processed/
ls
ln -s edn_subject1_train20 ~/poseVideo/custom-diffusion/data/
xsync edn_subject1_train20 -remote zhicai.12
ls
mv edn_subject1_train20 edn/
ls
cd edn
ls
cd ..
mv edn edn_subject1_train20 
mkdir edn
mv edn_subject1_train20 edn/
ls
cd edn
ls
cd ..
mv edn_subject1_train40 edn/
mv edn_subject2_train40 edn/
mv edn_subject1_train20 edn/
ls
mv edn_subject2_val edn/
mkdir DFpair
xsync edn/ -remote zhicai.12 
ln -s edn/ ~/poseVideo/custom-diffusion/data/
ln -s edn/ ~/poseVideo/custom-diffusion/data
ln -s edn ~/poseVideo/custom-diffusion/data
ln -s edn ~/poseVideo/custom-diffusion/data/
cd edn/
ls
cd ..
ln -s edn ~/poseVideo/custom-diffusion/data
ln -s 
ln -s edn ~/poseVideo/custom-diffusion/data
ls
ln -s edn/ ~/poseVideo/custom-diffusion/data
cd edn/
ls
ls -a
ls -l
ls -ll
du -h
ls
cd ..
ln -s edn/ /home/zhicai/poseVideo/custom-diffusion/data
ln -s edn/ /home/zhicai/poseVideo/custom-diffusion/data/
ln -s edn/ /home/zhicai/poseVideo/custom-diffusion/data
pwd
python src/preprocess_pose_data.py
gpustat
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "photo of a woman, best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing pose, modern dressing"
conda activate ldm
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "photo of a woman, best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing pose, modern dressing"
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic5 woman configs/custom-diffusion/finetune_addtoken.yaml  models/v1-5-pruned-emaonly.safetensors "photo of a woman, best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing pose, modern dressing"
gpustat
ssh zhicai.12
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
gpustat
killall -9 python3
killall -9 python
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
conda activate ldm
ls
gpustat
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50
conda activate ldm
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50
num=2
$((num+1))
echo $((num+1))
echo $(($num+1))
_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 4
gpustat
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir /data/zhicai/dataset/dancedata/edn/subject1/val/primitives/processed/2Dpose --gen_video
python src/preprocess_pose_data.py
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir /data/zhicai/dataset/dancedata/edn/subject1/val/primitives/processed/2Dpose --gen_video
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject1/val_2Dpose --gen_video
CUDA_VISIBLE_DEVICES=3 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-11T13-26-24_man-sdv4/checkpoints/epoch=000008.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 50 --batch_size 2 --pose_dir data/edn_subject1/val_2Dpose --gen_video
gpustat
cdgpustat
gpustat
cdgpustat
gpustat
killall -9 python
gpustat
cd /data/zhicai/dataset/dancedata/edn/subject1
ls
cd ..
cd subject2
ls
cd val/
ls
cd test_
cd test_img/
ls
cd ..
pwd
conda activate ldm
gpustat
conda activate ldm
gpustat
bash scripts/sample_gen.sh
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic3 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic2 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth
checkput 1 41000 | python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> woman, a realistic person, a complete torso, short sleeves, colored picture, high resolution, simple background, long pants, dancing" --delta_ckpt logs/2023-03-19T15-00-35_woman-sdv4/checkpoints/epoch=000018.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject2_val/pose --scale 4 --grid
checkput 1 41000 | python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> woman, a realistic person, a complete torso, short sleeves, colored picture, high resolution, simple background, long pants, dancing" --delta_ckpt logs/2023-03-19T15-00-35_woman-sdv4/checkpoints/epoch=000018.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject2_val/pose --scale 4 --mode 'grid'
bash scripts/sample.sh
gpustat
src/preprocess_pose_data.py
python src/preprocess_pose_data.py
conda activate ldm
python src/preprocess_pose_data.py
du -h
mv -r logs/ /data/zhicai/logs/custom/I
mv logs/ /data/zhicai/logs/custom/
ln -s /data/zhicai/logs/custom/ ./logs
bash scripts/finetune_gen.sh "man" data/edn_subject1_train20 gen_reg/samples_man man configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
bash scripts/finetune_gen.sh "man" data/edn_subject1_train20 gen_reg/edn_subject1_train20 man configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
gpustat
CUDA_VISIBLE_DEVICES=5 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-15T17-30-07_man-sdv4/checkpoints/epoch=000005.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject1/val_2Dpose --scale 4 --gen_video
gpustat
CUDA_VISIBLE_DEVICES=4 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> man" --delta_ckpt logs/2023-03-15T17-30-07_man-sdv4/checkpoints/epoch=000005.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject1/val_2Dpose --scale 4 --gen_video
CUDA_VISIBLE_DEVICES=5 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> woman" --delta_ckpt /home/zhicai/poseVideo/custom-diffusion/logs/2023-03-13T11-06-43_woman-sdv4/checkpoints/epoch=000005.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject2_val/pose --scale 4 --gen_video
gpustat
CUDA_VISIBLE_DEVICES=4 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> woman" --delta_ckpt /home/zhicai/poseVideo/custom-diffusion/logs/2023-03-13T11-06-43_woman-sdv4/checkpoints/epoch=000005.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject2_val/pose --scale 4 --gen_videodd
CUDA_VISIBLE_DEVICES=4 python sample_pose.py --config configs/cldm/finetune_addtoken.yaml --prompt "photo of a <new1> woman, a realistic person with a complete torso and a weak light and shadow effect on the top right corner" --delta_ckpt /home/zhicai/poseVideo/custom-diffusion/logs/2023-03-13T11-06-43_woman-sdv4/checkpoints/epoch=000005.ckpt --ckpt models/ControlNet/control_sd15_openpose.pth --ddim_steps 50 --n_samples 20 --batch_size 2 --pose_dir data/edn_subject2_val/pose --scale 4 --gen_video
sh scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
code ~/bin/check_gpu 
check_gpu 2 30000
check_gpu 2 300000
check_gpu 2 30000
check_gpu 2 3000
gpustat
check_gpu 2 20000
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_realistic_train20 woman configs/cldm/finetune_addtoken.yaml  models/ControlNet/control_sd15_openpose.pth
gpustat
xsync ~/bin/ ./ -remote zhicai.12
pip list
cd data
ls
cd data
rm data
ls
ln -s /data/zhicai/dataset/dancedata/processed/edn/ data/
ln -s /data/zhicai/dataset/dancedata/processed/edn/ data
xsync ./ -remote zhicai
xsync ./ -remote zhicai.12
conda activate ldm
gpustat
bash scripts/sample.sh 
conda activate ldm
bash scripts/sample.sh 
xsync -remote zhicai.12
xsync ./ -remote zhicai.12
gpustat
xsync ./ -remote zhicai.12
ls -l data
xsync /data/zhicai/dataset/dancedata/processed/edn/solo/ -remote zhicai.12
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
conda activate ldm
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no-testdd
gpustat
cd //data/zhicai/dataset
ls
cd dancedata/
ls
cd solo/
ls
gpustat
pwd
gpustat
bash scripts/sample.sh 
src/preprocess_pose_data.py
python src/preprocess_pose_data.py
bash scripts/sample.sh
bash scripts/sample.sh 
bash scripts/sample.sh
conda activate ldm
/data/zhicai/miniconda3/envs/ldm/bin/python
python sample_script.py
conda activate ldm
gpustat
conda activate ldm
gpustat
python sample_script.py 
gpustat
conda activate ldm
gpustat
python sample_script.py 
python sample_script.py
gpustat
ssh zhicai.12
conda activate ldm
bash scripts/sample_gen.sh "man" data/edn_subject1_train20 gen_reg/edn_subject1_train20_realistic1 man configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "photo of a man, best quality, ultra high res, (photorealistic:1.4), a complete torso, colored picture, high resolution, simple background, dancing, random pose"
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic3 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth
gpustat
bash scripts/sample.sh 
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic2 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing, random pose"
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing, random pose"
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
python src/preprocess_pose_data.py
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no_test
bash scripts/sample_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic4 woman configs/custom-diffusion/finetune_addtoken.yaml  models/realisticVisionV13_v13.safetensors "photo of a woman, best quality, ultra high res, (photorealistic:1.4), a complete torso,colored picture, high resolution, simple background, dancing, random pose"
conda activate ldm
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000014.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/2023-03-26T12-00-08_dog-sdv4/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000019.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
gpustat
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt --no-test
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog_genReg finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
gpustat
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog_realReg finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt 
gpustat
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a dog on the beach under the eiffel tower' --delta_ckpt
logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000009.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a dog on the beach under the eiffel tower' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000009.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a <new1> dog on the beach under the eiffel tower' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000009.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a <new1> dog under the eiffel tower' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000009.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
ils
ls
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken2.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
conda activate ldm
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken2.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
killall -9 python3
killall -9 python
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken2.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000004.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000014.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=3 python sample.py --prompt 'photo of a <new1> dog' --delta_ckpt logs/2023-03-26T12-52-38_dog-sdv4/checkpoints/epoch=000005.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
CUDA_VISIBLE_DEVICES=2 python sample.py --prompt 'photo of a dog on the beach' --delta_ckpt logs/2023-03-27T05-43-18_dog/checkpoints/epoch=000019.ckpt --ckpt Stable-diffusion/sd-v1-4-full-ema.ckpt --n_samples 8 --n_iter 2
gpustat
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog_realReg finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt --no-test
bash scripts/finetune_real.sh "dog" data/dog data/retrieve_dog_200  dog_realReg finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/finetune_gen.sh "dog" data/dog real_reg/samples_dog  dog_genReg finetune_addtoken.yaml Stable-diffusion/sd-v1-4-full-ema.ckpt
bash scripts/sample.sh 
conda activate ldm
bash scripts/sample.sh 
conda activate ldm
python sample_script.py 
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no-test
conda activate ldm
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no-test
gpustat
ssh zhicai.12
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no-test
gpustat
gpustt
gpustat
bash scripts/finetune_gen.sh "woman" data/edn_subject2_train20 gen_reg/edn_subject2_train20_realistic6 woman configs/cldm/finetune_addtoken.yaml   models/ControlNet/control_sd15_openpose.pth --no-test
gpustat
conda activate ldm
gpustat
git init
git add .gitignore 
git status
git commit -m 'ignore'
git config --global user.email "wangzhic@mail.ustc.edu.cn"
git commit -m 'ignore'
git log
git checkout master
git status
git log
git remote
git remote set-url origin https://github.com/Zhicaiwww/custom-diffusion
git add .gitignore 
git commit -m 'ignore'
t config --global user.name "zhicaiWWW"
git config --global user.name "zhicaiWWW"
git commit -m 'ignore'
git status
git add .
git staus
git status
git cache -rm outputs/*
git help
git rm outputs/
git rm -r outputs/
git rm -r outputs/*
git rm  outputs/*
git reset outputs/
git status
git commit -m "custom-v_only"
gpustat
ssh zhicai.12
mv logs/ /data/zhicai/logs/custom-diffusion
ln -s /data/zhicai/logs/custom ./logs
xsync ./ -remote zhicai.12
ln -s /data/zhicai/logs/custom-diffusion/ ./logs
gpustat
zsh
bash train_synkey_ablation.sh 
conda activat eldm
codna activate ldm
conda activat eldm
bash train_synkey_ablation.sh 

bash train_synkey_ablation.sh 
gpustat
python main.py --c configs/cub/cub_train_resnet50.yaml --dataset cub --synthetic-path --gpu_id 7 --exp_name basline
cd ../FGVC-PIM/
python main.py --c configs/cub/cub_train_resnet50.yaml --dataset cub --synthetic-path --gpu_id 7 --exp_name basline
python main.py --c configs/cub/cub_train_resnet50.yaml --dataset cub  --gpu_id 7 --exp_name basline
gpustat
wget goolge.com
clash
zsh
conda activate ldm
zsh
zsh
exit
bash
exit
echo $PATH
bash
:
outputs/scripts_daily/classification/classification_main.szsh
conda activate ldm
bash  outputs/scripts_daily/classification/classification_main.sh
gpustat
zsh
scripts/classification/classification_main.sh
zsh
/bin/python3 /home/zhicai/.vscode-server/extensions/ms-python.python-2023.22.1/pythonFiles/printEnvVariablesToFile.py /home/zhicai/.vscode-server/extensions/ms-python.python-2023.22.1/pythonFiles/deactivate/bash/envVars.txt
