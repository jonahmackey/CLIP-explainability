from dream import clip_dream, visualize_dream_loss

from datetime import datetime


img_lst = [
    "./Images/bananas.jpg",
    "./Images/bird.jpg",
    "./Images/crab.png",
    "./Images/david.jpg",
    "./Images/dog.jpg",
    "./Imaegs/street_scene.jpg",
    "./Images/two_bananas.jpg",
    "./Images/two_dogs.jpg",
    "./Images/xavier.jpg"
]

sv_index_lst = [
    0,
    1,
    2,
    3,
    4,
    5
]

threshold_lst = [
    0.1,
    0.05,
]

freq_reg_lst = [
    None,
    "norm"
]

lr_lst = [
    1,
    250,
    500,
    750
]

for img_fp in img_lst:
    i = 0
    for sv_index in sv_index_lst:
        for threshold in threshold_lst:
            for freq_reg in freq_reg_lst:
                for lr in lr_lst:
                    exp_name = img_fp[9:-4]
                    
                    save_path = "./Results/" + exp_name + f"{i}" # datetime.now().strftime("%y%m%d-%H%M%S-") + exp_name

                    clip_dream(img_fp=img_fp,
                                save_path=save_path,
                                theta=1,
                                sv_index=sv_index,
                                rand_dir=False,
                                threshold=threshold, 
                                eps=1e-6,
                                num_cutouts=32,
                                freq_reg=freq_reg,
                                lr=lr,
                                max_iters=5000)
                    
                    file = open(save_path + "/myfile.txt", "w")
                    params = f"Img encoder: {'ViT-B/32'}\n" \
                        + f"img_fp: {img_fp}\n" \
                        + f"save_path: {save_path}\n" \
                        + f"theta: {1}\n" \
                        + f"sv_index {sv_index}\n" \
                        + f"rand_dir: {False}\n" \
                        + f"threshold: {threshold}\n" \
                        + f"eps: {1e-6}\n" \
                        + f"num_cutouts: {32}\n" \
                        + f"freq_reg: {freq_reg}\n" \
                        + f"lr: {lr}\n" \
                        + f"max_iters: {3000}"
                    _ = file.write(params)
                    file.close()
                    
                    visualize_dream_loss(data_fp=save_path + "/experiment_data.csv", threshold=threshold, save_path=save_path)
                    
                    i += 1
                    