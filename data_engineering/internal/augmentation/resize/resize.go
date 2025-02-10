package resize

import (
	"fmt"
	"image"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

type Resize struct {
	width  int
	height int
}

func NewResize() augmentation.Augmentor {
	return &Resize{
		width:  224,
		height: 224,
	}
}

func (r *Resize) GetName() string {
	return "resized"
}

func (r *Resize) AugmentImage(img gocv.Mat) gocv.Mat {
	resized := gocv.Mat{}
	fmt.Println("Resizing img")

	gocv.Resize(img, &resized, image.Point{X: r.width, Y: r.height}, 0, 0, gocv.InterpolationCubic)
	return resized
}
