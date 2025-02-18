package augmentation

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/checkioname/cnn-skin-model/internal/utils"
	"gocv.io/x/gocv"
)

type Augmentor interface {
	AugmentImage(gocv.Mat) gocv.Mat
	GetName() string
}

func AugmentImages(img gocv.Mat, filters ...Augmentor) gocv.Mat {
	result := img.Clone()
	for _, filter := range filters {
		if filter.GetName() == "resized" {
			result = filter.AugmentImage(img)
		}
		result = filter.AugmentImage(result)
	}
	return result
}

// batch augmentation
func BatchAugmentation(inpDir, outDir string, filters ...Augmentor) {
  entries := utils.ReadNestedDir(inpDir)
	batches, _ := utils.GetBatches(entries, 2)

	var wg sync.WaitGroup

	for i, batch := range batches {
		wg.Add(1)
		fmt.Println("Starting augmentation on batch", i)
		go func(batchCopy []string) {
			defer wg.Done()

			for _, entry := range batchCopy {
				var outprefix string
				for _, f := range filters {
					outprefix += fmt.Sprintf("%v_", f.GetName())
				}
				outpath := filepath.Join(outDir, fmt.Sprintf("%v%v", outprefix, entry))

				outpath = strings.Replace(outpath, ".jpg", "_no_bg.jpg", 1)

				imgfullpath := filepath.Join(inpDir, entry)
				img := gocv.IMRead(imgfullpath, gocv.IMReadColor)
				if img.Empty() {
					fmt.Println("Image vazia - pulando", imgfullpath)
					continue
				}
				defer img.Close()

				result := AugmentImages(img, filters...)
				defer result.Close()
				if result.Empty() {
					fmt.Println("O resultado da transformacao eh vazio")
					break
				}
				gocv.IMWrite(outpath, result)
			}
		}(batch)
	}

	wg.Wait()
}
