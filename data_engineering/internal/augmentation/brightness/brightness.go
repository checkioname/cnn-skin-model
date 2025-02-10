package brightness

import (
	"fmt"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

// this code adjuste the brightness and contrast

type Brightness struct {
	contrast   float32
	brightness float32
}

func NewBrightness() augmentation.Augmentor {
	return &Brightness{
		contrast:   1.2,
		brightness: 20.0,
	}
}

func (b *Brightness) GetName() string {
	return "bright"
}

func (b *Brightness) AugmentImage(img gocv.Mat) gocv.Mat {
	fmt.Println("Brightening  img")

	adjustedImg := gocv.Mat{}

	img.ConvertToWithParams(&adjustedImg, img.Type(), b.contrast, b.brightness)
	return adjustedImg
}
