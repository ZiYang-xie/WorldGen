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
- **Instant 3D Generation** ⚡️ : Create full 3D scenes from input data in seconds
- **360° Free Exploration** 🧭 : WorldGen supports free 360° consistent exploration of the generated 3D scene with loop closure.
- **Diverse Scenes Support** 🌈 : WorldGen supports bothoor and outdoor scenes, both realistic and unrealistic scenes in any style.
- **Flexible Rendering** 📸 : WorldGen supports rendering at any resolution with any camera setting and trajectory in real-time.

Two lines of code to generate a 3D scene in seconds
```python
# Use our API to generate a 3D scene
worldgen = WorldGen()
worldgen.generate_world("<TEXT PROMPT to describe the scene>")
```

## Test-to-Scene Generatio
<div align="center">
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/text2scene/indoor1.gif" alt="demo" width="400"/>  
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/text2scene/outdoor1.gif" alt="demo" width="400"/>  
  <br>
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/text2scene/indoor2.gif" alt="demo" width="400"/>  
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/text2scene/outdoor2.gif" alt="demo" width="400"/>  
</div>

## Image-to-Scene Generation
<div align="center">
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/img2scene/painting.png" alt="demo" width=300 height=200 /> &nbsp;
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/img2scene/painting.gif" alt="demo" width=350 height=200/>  
  <br>
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/img2scene/street.png" alt="demo" width="300" height=200 /> &nbsp;
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/img2scene/street.gif" alt="demo" width=350 height=200/>  
</div>

## Free-viewpoint Exploration in 3D Scene
<div align="center" style="margin-bottom: 15px;">
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/free_explore/beach-converted.gif" alt="demo" width="400"/>  
  <img src="https://github.com/ZiYang-xie/WorldGen/blob/demos/assets/free_explore/indoor-converted.gif" alt="demo" width="400"/>  
  <br>
</div>

---

## News and TODOs
- [x] Opensource the WorldGen codebase 🎉
- [x] `04.17.2025` Add support for text-to-scene generation
- [x] `04.19.2025` Add support for image-to-scene generation
- [ ] Build a project page for WorldGen
- [ ] Support better img-to-scene generation (e.g., higher resolution, better lora training)
- [ ] Release huggingface demo.
- [ ] Support better background inpainting (Invisible region inpainting)
- [ ] High-resolution 3D scene generation

## 📦 Installation

Getting started with WorldGen is simple!

```bash
# Clone the repository 
git clone https://github.com/ZiYang-xie/WorldGen.git
cd WorldGen

# Create a new conda environment
conda create -n worldgen python=3.11
conda activate worldgen

# Install torch and torchvision
pip install torch torchvision

# Install worldgen
pip install .
```

## 🎮 Quick Start / Usage

Generate your first 3D scene in seconds, just need a few lines of code.   
We support two modes of generation:
- 📝 Generate a 3D scene from a text prompt 
- 🖼️ Generate a 3D scene from an image 

### WorldGen API
Quick start with WorldGen (mode in `t2s` or `i2s`):
```python
# Example using the Python API
from worldgen import WorldGen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📝 Generate a 3D scene from a text prompt
worldgen = WorldGen(mode="t2s", device=device)
splat = worldgen.generate_world("<TEXT PROMPT to describe the scene>")

# 🖼️ Generate a 3D scene from an image
worldgen = WorldGen(mode="i2s", device=device)
image = Image.open("path/to/your/image.jpg")
splat = worldgen.generate_world(
    image=image,
    prompt="<Optional: TEXT PROMPT to describe the image and the scene>",
)

# Save splat file as a .ply file, which can be load and visualized use standard gaussian splatting viewer
splat.save("path/to/your/output.ply")
```

> [!Tip]
> We also support background inpainting in for better scene generation, but it's currently an experimental feature, which may not work for all scenes.  
> It can be enabled by setting `WorldGen(inpaint_bg=True)`.
```bash
# If want to use background inpainting feature, install iopaint
pip install iopaint --no-dependencies
```


## 🕹️ [Recommended] Demo with 3D Scene Visualization
We provide a demo script to help you quickly get started and visualize the 3D scene in a web browser. The script is powered by [Viser](https://github.com/nerfstudio-project/viser).
```bash
# Generate a 3D scene from a text prompt
python demo.py -p "A beautiful landscape with a river and mountains"

# Generate a 3D scene from an image
python demo.py -i "path/to/your/image.jpg"

# You can also provide a text prompt to describe the image and the scene
python demo.py -i "path/to/your/image.jpg" -p "<TEXT PROMPT to describe the image and the scene>"
```
After running the demo script, A local viser server will be launched at `http://localhost:8080`, where you can explore the generated 3D scene in real-time.

---

> [!Note]
> **WorldGen** internally support generating a 3D scene from a 360° panorama image 📸, which related to how WorldGen works:
> You can try it out if you happen to have a 360° panorama (equirectangular) image. Aspect ratio of the panorama image should be 2:1.
```python
 pano_image = Image.open("path/to/your/pano_image.jpg")
 splat = worldgen._generate_world(pano_image=pano_image)
```

## 📚 Citation
If you find this project useful, please consider citing it as follows:
```bibtex
@misc{worldgen2025ziyangxie,
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
This project is built on top of the follows, please consider citing them if you find them useful:
- [Unik3D](https://github.com/lpiccinelli-eth/UniK3D)
- [Layerpano3D](https://github.com/3DTopia/LayerPano3D)
- [Viser](https://github.com/nerfstudio-project/viser)
- [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [LaMa](https://github.com/saic-mdal/lama)

This project is also inspired by the following projects:
- [WonderWorld](https://github.com/KovenYu/WonderWorld)
- [Differential Diffusion](https://github.com/exx8/differential-diffusion)
