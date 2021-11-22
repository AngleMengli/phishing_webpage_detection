from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE

if __name__ == '__main__':
    image = Image.open('dataset/train/absa/T0_3.png')  # 图片目录

    # 处理图像的一套流水线
    transform_train = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
     #   transforms.ToTensor(),
  #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 直接传入图像即可处理
    target_img = transform_train(image)  # 对图像做处理

    # 显示图片
    target_img.show()
    # 保存图片
    target_img.save('test.png')  # 保存