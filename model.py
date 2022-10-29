import torchvision
def get_model(device):
    # Load the model.
    model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)
    # Load the model onto the computation device.
    model = model.eval().to(device)
    return model