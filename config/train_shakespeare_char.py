# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such


CONFIG  = {
    "out_dir" : 'out-shakespeare-char'
    ,"eval_interval" : 250 
    ,"eval_iters" : 200
    ,"log_interval" : 50 
    ,"always_save_checkpoint" : False

    ,"wandb_log" : False # override via command line if you like
    ,"wandb_project" : 'shakespeare-char'
    ,"wandb_run_name" : 'mini-gpt'

    ,"dataset" : 'shakespeare_char'
    ,"gradient_accumulation_steps" : 1
    ,"batch_size" : 128
    ,"block_size" : 512 

    ,"n_layer" : 8
    ,"n_head" : 8
    ,"n_embd" : 256
    ,"dropout" : 0.1
    ,"bias": False

    ,"learning_rate" : 1e-3 # with baby networks can afford to go a bit higher
    ,"max_iters" : 50000
    ,"lr_decay_iters" : 5000 # make equal to max_iters usually
    ,"min_lr" : 1e-4 # learning_rate / 10 usually

    ,"warmup_iters" : 100 # not super necessary potentially
}
