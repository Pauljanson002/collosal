import numpy as np
#script = f'python main.py lr=0.001 epochs=1000 method=dds optimizer=sgd  wandb=online workspace=flamingo_test_dds lambda_guidance=1 lambda_rgb=0 text={} guidance_scale={}'

text = "peacock in water"

optimizer_choices = ["sgd","adan"]
workspace_prefix = "hpo"

# guidance_scale = np.random.uniform(1.0,100)
guidance_scale_choices = [7.5,10.0,100,1.0,2.0]
lambda_guidance_choices = [1.0,0.1,0.5,0.01,2.0,5.0,10.0]

for i in range(10):
    lr = np.random.choice([0.001,0.0001,0.00001,0.05,0.1,0.01,0.005])
    optimizer = np.random.choice(optimizer_choices)
    guidance_scale = np.random.choice(guidance_scale_choices)
    lambda_guidance = np.random.choice(lambda_guidance_choices)
    workspace = f"{workspace_prefix}_{optimizer}_lr{lr}_lambda{lambda_guidance}_guidance{guidance_scale}"
    script = f'python main.py lr={lr} epochs=1000 method=dds optimizer={optimizer}  wandb=online workspace={workspace} lambda_guidance={lambda_guidance} text="{text}" guidance_scale={guidance_scale}'
    print(script)
