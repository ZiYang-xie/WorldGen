# WorldGen: Generate Any 3D Scene in Seconds 
<div align="center">
  <img src="./assets/logo.png" alt="logo" width="300" style="margin-bottom: 210px;"/>  
</div>


<div align="center">
  
[![GitHub Stars](https://img.shields.io/github/stars/ZiYang-xie/WorldGen?style=social&label=Star&maxAge=2592000)](https://github.com/ZiYang-xie/WorldGen/stargazers/)
![Badge](https://img.shields.io/badge/version-v0.1.0-blue)
![Badge](https://img.shields.io/badge/license-Apache--2.0-green)

</div>

> Author 👨‍💻: [Ziyang Xie](https://ziyangxie.site/) | Contact Email 📧: [ziyangxie01@gmail.com](mailto:ziyangxie01@gmail.com)  
> Feel free to contact me for any questions or collaborations!

## 🌟 Introduction
🌏 **WorldGen** can generate 3D scenes in seconds from text prompts and images.  It is a powerful tool for creating 3D environments and scenes for games, simulations, robotics, and virtual reality applications.  
- ⚡️ **Instant 3D Generation**: Create full 3D scenes from input data in seconds
- 🧭 **360° Free Exploration**: WorldGen supports free 360° consistent exploration of the generated 3D scene with loop closure.
- 🌈 **Diverse Scenes Support**: WorldGen supports both indoor and outdoor scenes, both realistic and unrealistic scenes in any style.
- 📸 **Flexible Rendering**: WorldGen supports rendering at any resolution with any camera setting and trajectory in real-time.

## Test-to-Scene Generation
<div align="center">
  <img src="./assets/text2scene/indoor1.gif" alt="demo" width="400"/>  
  <img src="./assets/text2scene/outdoor1.gif" alt="demo" width="400"/>  
  <br>
  <img src="./assets/text2scene/indoor2.gif" alt="demo" width="400"/>  
  <img src="./assets/text2scene/outdoor2.gif" alt="demo" width="400"/>  
</div>

## Image-to-Scene Generation

---


## News and TODOs
- [ ] Opensource the WorldGen codebase 🎉
- [x] `04.17.2025` Add support for text-to-3D generation
- [ ] Add support for image-to-3D generation
- [ ] Build a project page for WorldGen
- [ ] Support better background inpainting (layered depth)
- [ ] High-resolution 3D scene generation

## 📦 Installation

Getting started with WorldGen is simple!

```bash
# Clone the repository 
git clone https://github.com/ZiYang-xie/WorldGen.git
cd WorldGen

# Install dependencies
pip install -e .
pip install iopaint --no-dependencies
```


## 🎮 Quick Start / Usage

Generate your first 3D scene in seconds, just need a few lines of code    
We support three modes of generation:
- 📝 Generate a 3D scene from a text prompt 
- 🖼️ Generate a 3D scene from an image 
- 📸 Generate a 3D scene from a panorama image 

### WorldGen API
```python
# Example using the Python API
from worldgen import WorldGen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
worldgen = WorldGen(device)

# Generate a 3D scene from a text prompt
splat = worldgen.generate_world("A beautiful landscape with a river and mountains")

# Generate a 3D scene from an image
image = Image.open("path/to/your/image.jpg")
splat = worldgen.generate_world(
    image=image, 
    text="<OPTIONAL: TEXT PROMPT to describe the scene>"
)

# Generate a 3D scene from a panorama image
pano_image = Image.open("path/to/your/pano_image.jpg")
splat = worldgen.generate_world(pano_image=pano_image)
```

> [!NOTE]
> We also support background inpainting in for better scene generation, but it's currently an experimental feature, which may not work for all scenes.  
> It can be enabled by setting `WorldGen(..., inpaint_bg=True)`.


### Demo with 3D Scene Visualization
We provide a demo script to help you quickly get started and visualize the 3D scene in a web browser. The script is powered by [Viser](https://github.com/nerfstudio-project/viser).
```bash
# Generate a 3D scene from a text prompt
python demo.py -p "A beautiful landscape with a river and mountains"

# Generate a 3D scene from an image
python demo.py -i "path/to/your/image.jpg"

# Generate a 3D scene from a panorama image
python demo.py --pano "path/to/your/pano_image.jpg"
```



## 📚 Citation
If you find this project useful, please consider citing it as follows:
```bibtex
@misc{worldgen,
  author = {Ziyang Xie},
  title = {WorldGen: Generate Any 3D Scene in Seconds},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ZiYang-xie/WorldGen}},
}
```

---

## 🤝 Acknowledgements
This project is built on top of the follows:
- [Unik3D](https://github.com/lpiccinelli-eth/UniK3D)
- [Layerpano3D](https://github.com/3DTopia/LayerPano3D)
- [Viser](https://github.com/nerfstudio-project/viser)
- [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [LaMa](https://github.com/saic-mdal/lama)

