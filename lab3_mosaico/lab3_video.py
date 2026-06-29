import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 10

cap = cv2.VideoCapture(0)
ca1 = cv2.VideoCapture(1)

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)*2   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
# Define Video Frame Rate in fps
fps = 10.0

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('saida_4.avi', fourcc, fps, (int(width),int(height*2)) )

if not cap.isOpened():
    print("Cannot open camera")
    exit()
if not ca1.isOpened():
    print("Cannot open camera")
    exit()
    
print('entrando no while')
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret1, frame1 = ca1.read()
    # if frame is read correctly ret is True
    print('definições corretas')
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if not ret1:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    print('pegou imagem')

    img1 = frame
    img2 = frame1

    if img1 is None or img2 is None:
        raise FileNotFoundError("Verifique se as imagens próprias foram salvas corretamente no diretório.")

	# Conversão para escala de cinza para os extratores de feições
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# =====================================================================
	# PASSO 1: DETECÇÃO DE CARACTERÍSTICAS E CORRESPONDÊNCIA (MATCHING)
	# =====================================================================
	# Utilizaremos o SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

	# Força bruta para encontrar os melhores matches usando a razão de Lowe
    bf = cv2.BFMatcher()
    matches_brutos = bf.knnMatch(des1, des2, k=2)

	# Aplicação do teste de razão de Lowe para filtrar matches muito ambíguos
    bons_matches = []
    for m, n in matches_brutos:
        if m.distance < 0.75 * n.distance:
            bons_matches.append(m)

	# Extração das coordenadas numéricas dos pontos correspondentes
    src_pts = np.float32([kp1[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

    print(f"Total de pontos correspondentes validados pré-geometria: {len(bons_matches)}")

	# =====================================================================
	# PASSO 2: ALINHAMENTO 2D VIA MÍNIMOS QUADRADOS TRADICIONAIS (SEM RANSAC)
	# =====================================================================
	# O método clássico tenta ajustar todos os pontos de uma vez (method=0)
    H_ls, status_ls = cv2.findHomography(src_pts, dst_pts, method=0)

    print("\n--- Matriz de Homografia via Mínimos Quadrados Clássicos ---")
    print(H_ls)

	# =====================================================================
	# PASSO 3: MÍNIMOS QUADRADOS ROBUSTOS VIA RANSAC
	# =====================================================================
	# Definimos um limiar de reprojeção de 5 pixels para considerar um inlier
    reproj_threshold = 5.0
    H_ransac, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)

    print("\n--- Matriz de Homografia via RANSAC ---")
    print(H_ransac)

	# Contagem de Inliers e Outliers
    inliers_count = np.sum(mask_ransac)
    outliers_count = len(mask_ransac) - inliers_count
    print(f"\nResultado RANSAC: {inliers_count} Inliers | {outliers_count} Outliers")

	# Visualização das correspondências aceitas pelo RANSAC
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, bons_matches, None, 
                                  matchColor=(0, 255, 0), singlePointColor=None, 
                                    matchesMask=mask_ransac.ravel().tolist(), flags=2)

	#plt.figure(figsize=(12, 6))
	#plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
	#plt.title("Correspondências (Inliers) Selecionadas pelo RANSAC")
	#plt.axis('off')
	#plt.show()

	# =====================================================================
	# PASSO 4: COSTURA DE IMAGENS (IMAGE STITCHING)
	# =====================================================================
	# Vamos rotacionar e projetar a Imagem 1 no plano da Imagem 2
    width = img1.shape[1] + img2.shape[1]
    height = max(img1.shape[0], img2.shape[0])

	# Projeção usando Homografia por Mínimos Quadrados
    img1_warped_ls = cv2.warpPerspective(img1, H_ls, (width, height))

	# Projeção usando Homografia pelo RANSAC
    img1_warped_ransac = cv2.warpPerspective(img1, H_ransac, (width, height))

	# Criação do Mosaico (colando a imagem 2 por cima da zona de projeção)
    mosaico_ls = img1_warped_ls.copy()
    mosaico_ls[0:img2.shape[0], 0:img2.shape[1]] = img2

    mosaico_ransac = img1_warped_ransac.copy()
    mosaico_ransac[0:img2.shape[0], 0:img2.shape[1]] = img2

	# Exibição dos resultados comparativos
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    frame_out = cv2.cvtColor(mosaico_ls, cv2.COLOR_BGR2RGB)
    frame_out1 = cv2.cvtColor(mosaico_ransac, cv2.COLOR_BGR2RGB)
	#ax[0].imshow(cv2.cvtColor(mosaico_ls, cv2.COLOR_BGR2RGB))
	#ax[0].set_title("Mosaico Final - Mínimos Quadrados Tradicionais (Pode falhar se houver outliers)")
	#ax[0].axis('off')

	#ax[1].imshow(cv2.cvtColor(mosaico_ransac, cv2.COLOR_BGR2RGB))
	#ax[1].set_title("Mosaico Final - RANSAC Robusto")
	#ax[1].axis('off')
	
    cv2.imshow('frame',cv2.vconcat([frame_out, frame_out1]))
    out.write(cv2.vconcat([frame_out, frame_out1]))
	
    if cv2.waitKey(1) == ord('q'):
        break

	#plt.tight_layout()
	#plt.show()
	
cap.release()
ca1.release()
cv2.destroyAllWindows()
