attack.py 是攻击代码，目前已经实现的攻击方式有
            "obs_grd_single", 
            "obs_noise_single",
            "obs_grd_all",
            "obs_noise_all",
            "act_greedy_single",
            "act_noise_single",
            "act_greedy_all",
            "act_noise_all"
其中 obs_grd_single 表示对单个智能体的观测进行白盒梯度攻击，obs_noise_single 表示对单个智能体的观测进行噪声攻击，obs_grd_all 表示对所有智能体的观测进行白盒梯度攻击，obs_noise_all 表示对所有智能体的观测进行噪声攻击，act_greedy_single 表示对单个智能体的动作进行贪婪攻击，act_noise_single 表示对单个智能体的动作进行噪声攻击，act_greedy_all 表示对所有智能体的动作进行贪婪攻击，act_noise_all 表示对所有智能体的动作进行噪声攻击。

attack.py 使用的runner是casia_fuzzy_attack/EGH-MARL-play/harl/runners/on_policy_ma_eval_attack_runner.py

play.py 是多智能体系统使用已经训练的权重进行重演的代码
使用的runner是casia_fuzzy_attack/EGH-MARL-play/harl/runners/on_policy_ma_eval_play.py
这个runner会返回一个log列表，play.py 会根据这个log列表进行文件输出，文件名是 test_log.txt

plot_test.py 是使用 test_log.txt 文件进行策略可视化的代码
由于要体现对称性，已经把使用策略输出值的绝对值进行绘制
