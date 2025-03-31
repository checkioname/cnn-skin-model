package crop

import (
	"fmt"
	"image"

	"github.com/checkioname/cnn-skin-model/internal/augmentation"
	"gocv.io/x/gocv"
)

type Crop struct {
	region string // center, random
	size   int    // 224 (its always gonna be a square)
	number int    // only apply for random crop

}

func NewCrop(r string, s int, n int) augmentation.Augmentor {
	return &Crop{
		region: r,
		size:   s,
		number: n,
	}
}

func (c *Crop) GetName() string {
	return "crop"
}

func (c *Crop) AugmentImage(img gocv.Mat) gocv.Mat {
	copyimg := img.Clone()

	imgheight := img.Rows()
	imgwidth := img.Cols()

	fmt.Println(imgheight)
	fmt.Println(imgwidth)

	x1 := (imgheight - c.size) / 2
	x2 := x1 + c.size

	y1 := (imgwidth - c.size) / 2
	y2 := y1 + c.size

	//Region para cortar a imagem (precisa passar rect da imagem)
	croprect := image.Rect(x1, y1, x2, y2)
	croppedimg := copyimg.Region(croprect)

	path := "/home/king/Documents/PsoriasisEngineering/cnn-skin-model/data_engineering/internal/augmentation/crop/test3.png"
	gocv.IMWrite(path, croppedimg)

	fmt.Println(img)
	return img
}
