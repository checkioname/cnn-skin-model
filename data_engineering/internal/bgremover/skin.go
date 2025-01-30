package bgremover

import "gocv.io/x/gocv"


func SegmentateSkin(path string) {
  
  img := gocv.IMRead(path, gocv.IMReadColor)
  if img.Empty() {
    panic("A imagem eh vazia")
  }
  defer img.Close()


  // converter o dominio de cores da imagem
  ycbrImg := gocv.NewMat()
  defer ycbrImg.Close()

  gocv.CvtColor(img, &ycbrImg, gocv.ColorRGBToYCrCb)


  // separar os canais da imagem para filtro
  channels := gocv.Split(ycbrImg)
  y, cb, cr := channels[0], channels[1], channels[2] 
  defer y.Close()
  defer cb.Close()
  defer cr.Close()


  // criar nnovas mascaras para valor de cb e cr
  cbMask := gocv.NewMat()
  crMask := gocv.NewMat()
  defer cbMask.Close()
  defer crMask.Close()

  gocv.InRangeWithScalar(cb, gocv.NewScalar(100, 0, 0, 0), gocv.NewScalar(150, 0, 0, 0), &cbMask)
	gocv.InRangeWithScalar(cr, gocv.NewScalar(150, 0, 0, 0), gocv.NewScalar(200, 0, 0, 0), &crMask)

	// Combinar as máscaras de Cb e Cr para identificar a pele
	skinMask := gocv.NewMat()
	defer skinMask.Close()
	gocv.BitwiseAnd(cbMask, crMask, &skinMask)

	// Aplicar a máscara na imagem original
	result := gocv.NewMat()
	defer result.Close()
	gocv.BitwiseAndWithMask(img, img, &result, skinMask)

	// Salvar o resultado
	outputPath := "output_skin_detection.png"
	gocv.IMWrite(outputPath, result)

}
