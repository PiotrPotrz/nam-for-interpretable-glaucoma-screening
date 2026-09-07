import timm

avail_pretrained_models = timm.list_models(pretrained=False)
print("Available pretrained models: ", len(avail_pretrained_models))
with open("timm_models_list_no.txt", "w") as f:
    for model in avail_pretrained_models:
        f.writelines(model + "\n")