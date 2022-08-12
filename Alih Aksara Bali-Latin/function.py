from heapq import *

def identification(returnFix, pathImage):
  import glob
  import cv2 as cv
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage.filters import sobel
  from skimage.filters import threshold_otsu
  from skimage.util import invert
  # SET DIRECTORY
  dir = 'D:/1808561016-Alihaksara Bali/FILE LAPORAN/SadaTA/data'

  def gray2biner(img):
      thr = 160
      im_gray = img
      im_bool = im_gray > thr
      maxval = 255
      im_bin = (im_gray > thr) * maxval
      return im_bin

  def horizontal_projections(sobel_image):
      return np.sum(sobel_image, axis=1)  


  def find_peak_regions(hpp, divider=2):
      threshold = (np.max(hpp)-np.min(hpp))/divider
      peaks = []
      peaks_index = []
      for i, hppv in enumerate(hpp):
          if hppv < threshold:
              peaks.append([i, hppv])
      return peaks

  #group the peaks into walking windows


  def get_hpp_walking_regions(peaks_index):
      hpp_clusters = []
      cluster = []
      for index, value in enumerate(peaks_index):
          cluster.append(value)

          if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
              hpp_clusters.append(cluster)
              cluster = []

          #get the last cluster
          if index == len(peaks_index)-1:
              hpp_clusters.append(cluster)
              cluster = []

      return hpp_clusters

  def heuristic(a, b):
      return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

  def astar(array, start, goal):

      neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
      close_set = set()
      came_from = {}
      gscore = {start:0}
      fscore = {start:heuristic(start, goal)}
      oheap = []

      heappush(oheap, (fscore[start], start))
      
      while oheap:

          current = heappop(oheap)[1]

          if current == goal:
              data = []
              while current in came_from:
                  data.append(current)
                  current = came_from[current]
              return data

          close_set.add(current)
          for i, j in neighbors:
              neighbor = current[0] + i, current[1] + j            
              tentative_g_score = gscore[current] + heuristic(current, neighbor)
              if 0 <= neighbor[0] < array.shape[0]:
                  if 0 <= neighbor[1] < array.shape[1]:                
                      if array[neighbor[0]][neighbor[1]] == 1:
                          continue
                  else:
                      # array bound y walls
                      continue
              else:
                  # array bound x walls
                  continue
                  
              if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                  continue
                  
              if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                  came_from[neighbor] = current
                  gscore[neighbor] = tentative_g_score
                  fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                  heappush(oheap, (fscore[neighbor], neighbor))
                  
      return []

  def get_binary(gray):
      mean = np.mean(gray)
      if mean == 0.0 or mean == 1.0:
          return gray

      thresh = threshold_otsu(gray)
      binary = gray <= thresh
      binary = binary*1
      return binary

  def path_exists(window_image):
      #very basic check first then proceed to A* check
      if 0 in horizontal_projections(window_image):
          return True
      
      padded_window = np.zeros((window_image.shape[0],1))
      world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
      path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1])))
      if len(path) > 0:
          return True
      
      return False

  def get_road_block_regions(nmap):
      road_blocks = []
      needtobreak = False
      
      for col in range(nmap.shape[1]):
          start = col
          end = col+20
          if end > nmap.shape[1]-1:
              end = nmap.shape[1]-1
              needtobreak = True

          if path_exists(nmap[:, start:end]) == False:
              road_blocks.append(col)

          if needtobreak == True:
              break
              
      return road_blocks

  def group_the_road_blocks(road_blocks):
      #group the road blocks
      road_blocks_cluster_groups = []
      road_blocks_cluster = []
      size = len(road_blocks)
      for index, value in enumerate(road_blocks):
          road_blocks_cluster.append(value)
          if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
              road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
              road_blocks_cluster = []

          if index == size-1 and len(road_blocks_cluster) > 0:
              road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
              road_blocks_cluster = []

      return road_blocks_cluster_groups



  #MAIN LINE SEGMEN
  image_read = cv.imread(pathImage)
  H, W = image_read.shape[:2]
  gray = np.zeros((H, W), np.uint8)
  for i in range(H):
      for j in range(W):
          gray[i, j] = np.clip(0.299 * image_read[i, j, 0] + 0.587
                              * image_read[i, j, 1] + 0.114 * image_read[i, j, 2], 0, 255)
  res,thresh_img = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV) 
  thinned = cv.ximgproc.thinning(thresh_img)
  thinned = cv.bitwise_not(thinned)
  
  #gray2binary
  binary = gray2biner(gray)

  #horizontal_projection to image
  sobel_image = sobel(gray)
  hpp = horizontal_projections(sobel_image)

  #to know area segment
  peaks = find_peak_regions(hpp)

  peaks_index = np.array(peaks)[:, 0].astype(int)

  segmented_img = np.copy(gray)
  r, c = segmented_img.shape
  for ri in range(r):
      if ri in peaks_index:
          segmented_img[ri, :] = 0

  hpp_clusters = get_hpp_walking_regions(peaks_index)

  binary_image = get_binary(gray)

  for cluster_of_interest in hpp_clusters:
      nmap = binary_image[cluster_of_interest[0]
          :cluster_of_interest[len(cluster_of_interest)-1], :]
      road_blocks = get_road_block_regions(nmap)
      road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
      #create the doorways
      for index, road_blocks in enumerate(road_blocks_cluster_groups):
          window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
          binary_image[cluster_of_interest[0]:cluster_of_interest[len(
              cluster_of_interest)-1], :][:, road_blocks[0]: road_blocks[1]+10][int(window_image.shape[0]/2), :] *= 0

  #make line segments
  line_segments = []
  for i, cluster_of_interest in enumerate(hpp_clusters):
      nmap = binary_image[cluster_of_interest[0]
          :cluster_of_interest[len(cluster_of_interest)-1], :]
      path = np.array(
          astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2), nmap.shape[1]-1)))
      offset_from_top = cluster_of_interest[0]
      path[:, 0] += offset_from_top
      line_segments.append(path)

  offset_from_top = cluster_of_interest[0]
  line_count = len(line_segments)
  line_segments_new = []
  for x in range(line_count):
      A = line_segments[x]
      A = np.flipud(A)
      line_segments_new.append(A)

  #extract image and export
  line_count = len(line_segments_new)
  x, y = binary.shape
  for gb in range(line_count-1):
      coba1 = np.copy(thinned)
      for k in range(y-1):

          bar = line_segments_new[gb][k][0]
          kol = line_segments_new[gb][k][1]
          bar1 = line_segments_new[gb+1][k][0]
          kol1 = line_segments_new[gb+1][k][1]

          for i in range(bar-1):
              coba1[i][kol] = 255
          f = bar1+1
          for i in range(bar1, x):
              coba1[i][kol1] = 255

      batas_atas = np.amin(line_segments_new[gb], axis=0)
      ba = batas_atas[0]-5
      batas_bawah = np.amax(line_segments_new[gb+1], axis=0)
      bb = batas_bawah[0]+5
      result_line_segment = coba1[ba:bb, 0:y-1]

      cv.imwrite(dir+'/hasil_segmentasi_baris/imgSegment_'+str(gb+1).zfill(2)
                  +'.png', result_line_segment)

  #SEGMENTASI KARAKTER
  import glob, os
  import cv2 as cv
  import matplotlib.pyplot as plt
  from skimage.filters import sobel
  import numpy as np


  #FUNGSI SORTING
  def partition(arr, low, high):
      i = (low-1)         
      pivot = arr[high]     
  
      for j in range(low, high):
          if arr[j] <= pivot:
              i = i+1
              arr[i], arr[j] = arr[j], arr[i]
              coord[i], coord[j] = coord[j], coord[i]
              
      arr[i+1], arr[high] = arr[high], arr[i+1]
      coord[i+1], coord[high] = coord[high], coord[i+1]
      return (i+1)
  

  def quickSort(arr, low, high):
      if len(arr) == 1:
          return arr
      if low < high:
          pi = partition(arr, low, high)
          quickSort(arr, low, pi-1)
          quickSort(arr, pi+1, high)
          
  #FUNGSI HORIZONTAL PROJECTION & mencari wilayah tengah aksara
  def horizontal_projections(sobel_image):
      return np.sum(sobel_image, axis=1)  

  def find_peak_regions(hpp, divider=2):
      threshold = (np.max(hpp)-np.min(hpp))/divider
      peaks = []
      peaks_index = []
      for i, hppv in enumerate(hpp):
          if hppv < threshold:
              peaks.append([i, hppv])
      return peaks


  #MAIN SEGMENTASI KARAKTER
  os.chdir(dir+"/hasil_segmentasi_baris")

  for file in glob.glob("*.png"):
      #membaca file citra hasil segmentasi baris
      #print(str(file))
      file_name=str(os.path.splitext(file)[0])
      #print(file_name)
      
      
      
      img = cv.imread(file,0)
      gray = img
      #plt.figure(figsize = (20,20))
      #plt.imshow(img,cmap = "gray")
      #break
      
      #CCL OPENCV
      img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)[1]  # ensure binary
      connectivity = 4  
      output = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)
      
      #MENAMPUNG HASIL CCL
      num_labels = output[0]
      #num_labels
      labels = output[1]
      #labels
      stats = output[2]
      #stats
      
      coord = []
      num = 0
      for i in range(1, num_labels):
          num += 1
          if stats[i][2]<10 or stats[i][3]<10:
              continue
          else:
              coord.append([stats[i][0], stats[i][1], stats[i][2], stats[i][3], num])
      #coord
      
      
      #MEMBUAT ARRAY BANTUAN UNTUK PENGURUTAN HASIL CCL
      len_coord = len(coord) 
      prep_sort = []
      x_cor = []
      for i in range(len_coord):
          prep_sort.append(coord[i][0]+coord[i][1])
          x_cor.append(coord[i][0])
      #prep_sort
      
      #MELAKUKAN PRYEKSI HORIZONTAL
      sobel_image = sobel(gray)
      hpp = horizontal_projections(sobel_image)
      #plt.plot(hpp)
      #plt.show()
      
      peaks = find_peak_regions(hpp)

      peaks_index = np.array(peaks)[:,0].astype(int)

      segmented_img = np.copy(gray)
      r,c = segmented_img.shape
      for ri in range(r):
          if ri in peaks_index:
              segmented_img[ri, :] = 0

      #plt.figure(figsize=(20,20))
      #plt.imshow(segmented_img, cmap="gray")
      #plt.show()
      
      
      cek_cut = []
      len_peaks_index = len(peaks_index)
      for i in range(len_peaks_index-1):
          if peaks_index[i+1]-peaks_index[i] > 5:
              #print(peaks_index[i], peaks_index[i+1])
              cek_cut.append([peaks_index[i], peaks_index[i+1]])
      
      
      cen_atas = cek_cut[0][0]
      cen_bawah = cek_cut[len(cek_cut)-1][1]
      cut_aksara = cen_bawah-cen_atas
      center_aksara = cen_atas+(cut_aksara//2)

      
      #print( cut_aksara, center_aksara, cen_atas, cen_bawah)
      
      
      #MENCARI NILAI Y PALING RENDAH DAN PALING TINGGI UNTUK 
      max_y = 0
      for i in range(len_coord):
          if coord[i][1]+coord[i][3]>max_y:
              max_y = coord[i][1]+coord[i][3]

      min_y = max_y        
      for i in range(len_coord):
          if coord[i][1]<min_y:
              min_y = coord[i][1]

      #print(max_y, min_y) 
      
      
      #urutkan hasil segment
      n = len_coord
      quickSort(x_cor, 0, n-1)
      #print(coord)
      
      
      #MEMPERBAIKI URUTAN HASIL SEGMENTASI
      for j in range(len_coord-1):
          aks=0
          for i in range(len_coord-1):
              aks += 1
              if coord[i+1][0]>=coord[i][0] and coord[i+1][0]<=coord[i][0]+coord[i][2]:
                  if coord[i+1][1]< coord[i][1]:
                      if  ((coord[i][0]+coord[i][2])-(coord[i-1][0]+coord[i-1][2])) < ((coord[i+1][0]+coord[i+1][2])-(coord[i][0]+coord[i][2])) and coord[i][1]>coord[i-1][1]:
                          continue
                      elif (coord[i+1][1]-min_y) < (center_aksara-coord[i+1][1]) and (max_y-(coord[i+1][1]+coord[i+1][3])) < ((coord[i+1][1]+coord[i+1][3])-center_aksara) :
                          continue
                      else:
                          coord[i], coord[i+1] = coord[i+1], coord[i]  
                          
      aks=0
      for i in range(len_coord-1):
          aks += 1
          if coord[i+1][0]>=coord[i][0] and coord[i+1][0]<=coord[i][0]+coord[i][2]:
              if coord[i+1][1]< coord[i][1]:
                  if coord[i+1][1]+coord[i+1][3]<= center_aksara:
                      coord[i], coord[i+1] = coord[i+1], coord[i]
    

      #SEGMENTASI LANJUTAN UNTUK MEMISAH AKSARA YANG MASIH MENYATU DENGAN AKSARA LAINNYA
      aks=0
      i=0
      while i < len_coord:
          aks += 1
          x,y,w,h,l=coord[i]
          #print(x,y,w,h,l)
          if coord[i][3] > (cut_aksara*1.65):
              #print(aks) 
              #ketika aksara yang ukurannya melebihi, posisinya mepet atas
              if coord[i][1]+coord[i][3] < cen_atas:
                  i+=1
                  continue
              elif abs(coord[i][1]-min_y) < abs(cen_atas-coord[i][1]):
                  #print( aks, "mepet atas")
                  #mepet atas dan sampai mepet bawah
                  if abs(max_y-(coord[i][1]+coord[i][3])) < abs((coord[i][1]+coord[i][3])-cen_bawah):
                      #print( aks, "full atas bawah")
                      h_temp = cen_atas-(y-3)
                      coord[i] = [x,y,w,h_temp,l]

                      i+=1
                      len_coord+=1
                      y_temp = y+h_temp
                      h_temp = cut_aksara+3
                      coord.insert(i,[x,y_temp,w,h_temp,l])

                      i+=1
                      len_coord+=1
                      jm_h = y+h
                      y = y_temp+h_temp
                      h_temp = jm_h-y
                      coord.insert(i,[x,y,w,h_temp,l])


                  else :
                      #print( aks, "mepet atas, akhir aksara tengah")
                      h_temp = cen_atas-(y+3)
                      coord[i] = [x,y,w,h_temp,l]

                      i+=1
                      len_coord+=1
                      y_temp = y+h_temp
                      coord.insert(i,[x,y_temp,w,h-h_temp,l])

              #ketika aksara yang ukurannya melebihi,posisi aksara mepet dengan cen_atas
              elif abs(coord[i][1]-cen_atas) < abs(cen_bawah-coord[i][1]):
                  #print(aks, "mepet cen atas, akhir mepet bawah")
                  h_temp = cut_aksara+3
                  coord[i]=[x,y,w,h_temp,l]

                  i+=1
                  len_coord+=1
                  y_temp = y+h_temp
                  coord.insert(i,[x,y_temp,w,h-h_temp,l])
                  
              else:
                  i+=1
                  continue
          i+=1  
          
      
      
      #SIMPAN CITRA SEGMENTASI KARAKTER
      count = 0
      for cor in coord:
          seg = labels.copy()
          #plt.imshow(seg)
          count += 1
          [x,y,w,h,lab] = cor
          #s = np.zeros((y-min_y, w))
          #u = np.zeros((max_y-(y+h),w))
          t = seg[y:y+h,x:x+w]
          for i in range(len(t)):
              for j in range(len(t[i])):
                  if t[i][j] == lab:
                      t[i][j] = 255
                  else :
                      t[i][j]= 0
          
          #plt.figure(figsize=(20,20))
          #plt.imshow(t, cmap = "gray")
          #plt.show
          #resize citra karakter
          height, width = t.shape
          
          h_resize = 63
          resize= h_resize/height*100
          w_resize = round(resize/100*width)
          
          dim = (w_resize, h_resize)
          t = np.array(t, dtype='uint8')
          t = cv.resize(t, dim, interpolation = cv.INTER_AREA)
          
          for i in range(h_resize):
              for j in range(w_resize):
                  if t[i][j]!=0 and t[i][j]!=255:
                      t[i][j]=255
                      
                      
          #plt.figure(figsize=(20,20))
          #plt.imshow(t, cmap = "gray")
          #plt.show
          #break

          #menentukkan posisi aksara
          j_atas = abs(cen_atas-(y+h))
          j_tengah = abs(cen_bawah-(y+h))
          j_bawah = abs(max_y-(y+h))
          
          s = np.zeros((h_resize, w_resize))
          u = np.zeros((h_resize, w_resize))
          #print(j_atas, j_tengah, j_bawah)
          #print(file_name, count)
          if (j_bawah<j_tengah and j_bawah<j_atas) or (y >= center_aksara and y+h > (cen_bawah+cut_aksara//2-5)):
              #print("j_bawah")
              char= np.concatenate(( s, u, t), axis=0)
          elif j_atas<j_tengah and j_atas<j_bawah:
              #print("j_atas")
              char= np.concatenate(( t, u, s), axis=0)
          else :
              #print("j_tengah")
              char= np.concatenate(( s, t, u), axis=0)
              
          #print(count)
          #print("===========\n\n")
          #plt.figure(figsize=(20,20))
          #plt.imshow(char, cmap = "gray")
          #plt.show
          #break
          cv.imwrite(dir+"/hasil_segmentasi_karakter/"+str(file_name)+"_"+str(count).zfill(2)+".png",char) 
          #break           #char 

      #break

  #EKSTRAKSI FITUR
  import cv2 as cv
  import numpy as np
  import matplotlib.pyplot as plt
  from math import copysign, log10
  import math
  import glob, os

  def calculateDistance(x1,y1,x2,y2):
      dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
      return dist

  #MAIN EKTRAKSI FITUR
  os.chdir(dir+"/hasil_segmentasi_karakter")
  name_file=[]
  label=[]
  for file in glob.glob("*.png"):
      #membaca file citra hasil segmentasi baris
      #print(str(file))
      #name_img = str(file)
      #print(name_img)
      
      file_name=str(os.path.splitext(file)[0])
      name_file.append(file_name)
      
  name_file.sort()
  #label.sort()
  #print(name_file)


  path = dir+'/hasil_segmentasi_karakter/'
  eks_fit = []
  jml_eks = 0
  for i in name_file:
      
      image = path+i+".png"
      file_name = i+".png"
      lbl = file_name.split("-")
      lbl = lbl[0]
      #print(label)
    
      #print(image)
      #print(file_name)
      img = cv.imread(image,cv.IMREAD_GRAYSCALE)
      ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)
      #plt.figure(figsize=(10,10))
      #plt.imshow(img, cmap = "gray")
      #plt.show
      h,w =img.shape
      #print(h,w)
      
      
      # resize image
      width_resize = w
      while width_resize % 9 != 0:
          width_resize += 1
      
      
      height_resize = 189
      dim = (width_resize, height_resize)
      

      resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

      #print('Resized Dimensions : ',resized.shape)
      #plt.figure(figsize=(10,10))
      #plt.imshow(resized, cmap = "gray")
      #plt.show
      
      #Mengubah citra menjadi biner (0 dan 1)
      h_resized, w_resized = resized.shape
      #print(h_resized, w_resized)
      for i in range(height_resize):
          for j in range(width_resize):
              if resized[i][j]!=0:
                  resized[i][j]=1
      
      #plt.figure(figsize=(10,10))
      #plt.imshow(resized, cmap = "gray")
      #plt.show
      
      
      #####################
      #  MOMEN INVARIANT  #
      #####################
      
      # Calculate Moments 
      moments = cv.moments(resized) 
      # Calculate Hu Moments 
      huMoments = cv.HuMoments(moments)
      
      # Log scale hu moments 
      for i in range(0,7):
          huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
      #print(huMoments)
      
      #Menampung 7 fitur invarian momment
      mo_inv = []
      for i in range(0,7):
          #print(huMoments[i][0])
          mo_inv.append(huMoments[i][0])
          
      #print(mo_inv)
      #print("\n\n")
      
      #####################
      #     ICZ & ZCZ     #
      #####################
      
      split_h = int(height_resize/9)
      split_w = int(width_resize/9)
      #ICZ
      #mencari centroid citra masukkan
      xp = 0
      yp = 0
      p = 0
      xc = 0
      yc = 0
      for i in range(height_resize):
          for j in range(width_resize):
              if resized[i][j]==1:
                  xp = xp+i
                  yp = yp+j
                  p += 1

      #print(xp, yp, p)
      xc = xp/p
      yc = yp/p
      xc = round(xc, 3)
      yc = round(yc, 3)
      #print(xc, yc)
      
      zone = 0
      tot_dist = 0
      jml_dist = 0
      icz = []
      batas_h = split_h
      batas_w = split_w
      x = 0
      y = 0
      x_akhir = batas_h
      y_akhir = batas_w
      while True:
          for i in range(x,x_akhir):
              for j in range(y,y_akhir):
                  if resized[i][j]==1:
                      cal_dist = calculateDistance(xc,yc,i,j)
                      tot_dist = tot_dist + cal_dist
                      jml_dist += 1
                      #print(i,j, resized[i,j])

          #print("----")
          y += batas_w
          y_akhir += batas_w
          zone += 1
          #print(y, y_akhir)

          if tot_dist == 0:
              tot_dist = 1

          if jml_dist == 0:
              jml_dist = 1
          feature = tot_dist/jml_dist
          #print(tot_dist)
          #print(jml_dist)
          #print(feature)
          icz.append(feature)
          tot_dist = 0
          jml_dist = 0
          feature = 0
          #print("reset")
          #print(tot_dist)
          #print(jml_dist)
          #print(feature)
          #print(icz)

          #print("ZONE ke-",zone)
          #print("----")
          #break

          #print(x_akhir-1, width_resize-1)

          if i == x_akhir-1 and j == width_resize-1:
              x += batas_h
              x_akhir += batas_h
              y = 0
              y_akhir = batas_w

          if i == height_resize-1 and j == width_resize-1:
              break
      #print(icz)
      #break
      #mencari centroid masing-masing zone
      zone = 0
      batas_h = split_h
      batas_w = split_w
      x = 0
      y = 0
      x_akhir = batas_h
      y_akhir = batas_w

      xp = 0
      yp = 0
      p = 0
      xc = 0
      yc = 0
      cen_zcz = []
      while True:
          for i in range(x,x_akhir):
              for j in range(y,y_akhir):
                  #print(i,j, resized[i,j])
                  if resized[i][j]==1:
                      xp = xp+i
                      yp = yp+j
                      p += 1
          
          if xp == 0:
              xp = 1
          if yp == 0:
              yp = 1
          if p == 0:
              p = 1
          xc = xp/p
          yc = yp/p
          xc = round(xc, 3)
          yc = round(yc, 3)
          #print(xc, yc)
          cen_zcz.append([xc, yc])

          #print("----")
          y += batas_w
          y_akhir += batas_w
          zone += 1
          #print(y, y_akhir)
          #print(jml_satu)
          #print("ZONE ke-",zone)
          #print("----")

          #print(x_akhir-1, width_resize-1)
          if i == x_akhir-1 and j == width_resize-1:
              x += batas_h
              x_akhir += batas_h
              y = 0
              y_akhir = batas_w

          if i == height_resize-1 and j == width_resize-1:
              break
      #print(cen_zcz)
      
      #ZCZ
      zone = 0
      batas_h = split_h
      batas_w = split_w
      x = 0
      y = 0
      x_akhir = batas_h
      y_akhir = batas_w

      tot_dist = 0
      jml_dist = 0
      zcz = []
      while True:
          for i in range(x,x_akhir):
              for j in range(y,y_akhir):
                  #print(i,j, resized[i,j])
                  if resized[i,j] == 1:
                      xc, yc = cen_zcz[zone]
                      cal_dist = calculateDistance(xc,yc,i,j)
                      tot_dist = tot_dist + cal_dist
                      jml_dist += 1

          if tot_dist == 0:
              tot_dist = 1

          if jml_dist == 0:
              jml_dist = 1

          feature = tot_dist/jml_dist
          #print(tot_dist)
          #print(jml_dist)
          #print(feature)
          zcz.append(feature)
          tot_dist = 0
          jml_dist = 0
          tot_dist = 0
          jml_dist = 0

          #print("----")
          y += batas_w
          y_akhir += batas_w
          zone += 1
          #print(y, y_akhir)
          #print(jml_satu)
          #print("ZONE ke-",zone)
          #print("----")
          jml_satu= 0
          #print(x_akhir-1, width_resize-1)
          if i == x_akhir-1 and j == width_resize-1:
              x += batas_h
              x_akhir += batas_h
              y = 0
              y_akhir = batas_w

          if i == height_resize-1 and j == width_resize-1:
              break
      #print(zcz)
      
      
      f_name = []
      f_name.append(file_name)
      label = []
      label.append(lbl)
      #print(f_name)
      #simpan hasil ekstraksi fitur  
      arr = f_name + mo_inv + icz + zcz + label
      #arr = mo_inv + icz + zcz
      #print(arr)
      
      #print(jml_eks)
      
      eks_fit.append(arr)
      
      
      #print(eks_fit)
      #print("\n\n\n")
      #break
      
  import pandas as pd

  df = pd.DataFrame(eks_fit, columns =['FName', 'mo_inv1', 'mo_inv2', 'mo_inv3', 'mo_inv4', 'mo_inv5', 'mo_inv6', 'mo_inv7', 'icz1', 'icz2', 'icz3', 'icz4', 'icz5', 'icz6', 'icz7', 'icz8', 'icz9', 'icz10', 'icz11', 'icz12', 'icz13', 'icz14', 'icz15', 'icz16', 'icz17', 'icz18','icz19', 'icz20', 'icz21', 'icz22', 'icz23', 'icz24', 'icz25', 'icz26', 'icz27','icz28', 'icz29', 'icz30', 'icz31', 'icz32', 'icz33', 'icz34', 'icz35', 'icz36','icz37', 'icz38', 'icz39', 'icz40', 'icz41', 'icz42', 'icz43', 'icz44', 'icz45','icz46', 'icz47', 'icz48', 'icz49', 'icz50', 'icz51', 'icz52', 'icz53', 'icz54','icz55', 'icz56', 'icz57', 'icz58', 'icz59', 'icz60', 'icz61', 'icz62', 'icz63','icz64', 'icz65', 'icz66', 'icz67', 'icz68', 'icz69', 'icz70', 'icz71', 'icz72', 'icz73', 'icz74','icz75', 'icz76', 'icz77', 'icz78', 'icz79', 'icz80', 'icz81', 'zcz1', 'zcz2', 'zcz3', 'zcz4', 'zcz5', 'zcz6','zcz7', 'zcz8', 'zcz9', 'zcz10', 'zcz11', 'zcz12', 'zcz13', 'zcz14', 'zcz15','zcz16', 'zcz17', 'zcz18', 'zcz19', 'zcz20', 'zcz21', 'zcz22', 'zcz23', 'zcz24','zcz25', 'zcz26', 'zcz27', 'zcz28', 'zcz29', 'zcz30', 'zcz31', 'zcz32', 'zcz33','zcz34', 'zcz35', 'zcz36', 'zcz37', 'zcz38', 'zcz39', 'zcz40', 'zcz41', 'zcz42','zcz43', 'zcz44', 'zcz45', 'zcz46', 'zcz47', 'zcz48', 'zcz49', 'zcz50', 'zcz51','zcz52', 'zcz53', 'zcz54', 'zcz55', 'zcz56', 'zcz57', 'zcz58', 'zcz59', 'zcz60','zcz61', 'zcz62', 'zcz63', 'zcz64', 'zcz65', 'zcz66', 'zcz67', 'zcz68', 'zcz69','zcz70', 'zcz71', 'zcz72', 'zcz73', 'zcz74', 'zcz75', 'zcz76', 'zcz77', 'zcz78','zcz79', 'zcz80', 'zcz81', 'label'], dtype = float)
  #df    
  df.to_csv(dir+'/fitur/features_aksara.csv', index=False)

  #PENGENALAN DAN RULE BASE
  import itertools
  import numpy
  import matplotlib.pyplot as plt
  import pandas
  import sklearn
  from sklearn.svm import SVC
  from sklearn.model_selection import train_test_split


  data_set=pandas.read_csv(dir+'/FEATURES EXTRACTION/features_aksara.csv', index_col=False)
  AKSARA =['adeg','ba_kembang_matedong','bisah','carik','cecek','da','da_madu','da_madu_matedong','ekor_bawah','ga','ga_matedong','gan_da','gan_da_madu','gan_ga','gan_ha','gan_na','gan_ra','gan_ta','gan_wa','gan_ya','gem_pa','ha','ha_matedong','ka','ka_matedong','la','ma','na','na_matedong','na-rambat','nga','pa','pepet','ra','sa','sa_sapa','suku','suku_ilut','surang','ta','taleng','tengah','ulu','wa','wa_matedong','ya']
  data_set=data_set[data_set.columns[1:171]]
  data_set[:].style

  x=data_set.iloc[:,:-1]
  y=data_set.iloc[:,-1]
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2,stratify=y)
  #print('Splitted Successfully')

  svm=SVC(kernel='rbf', C=1000,gamma=0.0001)
  svm.fit(x_train,y_train)

  data_test=pandas.read_csv(dir+'/fitur/features_aksara.csv', index_col=False)
  data_test=data_test[data_test.columns[1:171]]
  data_test[:].style

  x=data_test.iloc[:,:-1]
  y_pred=svm.predict(x)

  import pandas as pd

  df = pd.DataFrame(y_pred)
  #df
  df.to_csv(dir+'/hasil_pengenalan/aksara.csv', index=False)

  #RULE BASE
  import csv 

  aksara = []

  with open(dir+'/hasil_pengenalan/aksara.csv') as csvfile:
      csvReader = csv.reader(csvfile)
      for row in csvReader:
          aksara.append(row[0])        
  #print(aksara, len(aksara))

  del aksara[0]
  pjg_array = len(aksara)


  i=0
  while i < pjg_array:
      #print(i, aksara[i], pjg_array)
      
      if aksara[i] == 'ga_matedong':
          aksara[i] = 'ga'
          aksara.insert(i+1,'tedong')
          
      if aksara[i] == 'wa_matedong':
          aksara[i] = 'wa'
          aksara.insert(i+1,'tedong')
          
      if aksara[i] == 'na_matedong':
          aksara[i] = 'na'
          aksara.insert(i+1,'tedong')
          
          
      if aksara[i] == 'ka_matedong':
          aksara[i] = 'ka'
          aksara.insert(i+1,'tedong')
          
      if aksara[i] == 'ha_matedong':
          aksara[i] = 'ha'
          aksara.insert(i+1,'tedong')
        
      
      if aksara[i] == 'da_madu_matedong':
          aksara[i] = 'da_madu'
          aksara.insert(i+1,'tedong')
          
      if aksara[i] == 'ba_kembang_matedong':
          aksara[i] = 'ba_kembang'
          aksara.insert(i+1,'tedong')
          
      if aksara[i] == 'tengah' or aksara[i] == 'ekor_bawah':
          del aksara[i]
          i-=1
      
      if aksara[i] == 'suku_ilut' or aksara[i] == 'suku' :
          if aksara[i-1] == 'adeg' or aksara[i-1] == 'taleng':
              del aksara[i]
              i-=1
        
          
      if aksara[i] == 'ulu' or aksara[i] == 'ulu_sari' or aksara[i] == 'ulu_candra' or aksara[i] == 'ulu_ricem' or aksara[i] == 'cecek' or aksara[i] == 'surang' or aksara[i] == 'pepet':
          if aksara[i+1]=='gan_wa' or aksara[i+1]=='gan_da_madu' or aksara[i+1]=='gan_ra' or aksara[i+1]=='gan_ta'or aksara[i+1]=='gan_ha' or aksara[i+1]=='gan_ya' or aksara[i+1]=='gan_na' or aksara[i+1]=='gan_da' or aksara[i+1]=='suku' or aksara[i+1]=='suku_ilut':
              aksara[i], aksara[i-1] = aksara[i-1], aksara[i]
      
      if aksara[i] == 'bisah':
          if aksara[i-1] == 'ulu' or aksara[i-1] == 'ulu_sari' or aksara[i-1] == 'ulu_candra' or aksara[i-1] == 'ulu_ricem' or aksara[i-1] == 'cecek' or aksara[i-1] == 'surang' or aksara[i-1] == 'pepet':
              aksara[i], aksara[i-1] = aksara[i-1], aksara[i]
          
          
          
      pjg_array = len(aksara)
      i +=1
      

      
  i=0
  A=[]
  pjg_array=len(aksara)
  #pjg_array

  for j in range(pjg_array):
      #pengangge aksara
      if aksara[j]=='taleng'or aksara[j]=='taleng_detya' :
          A.append("taleng")
      elif aksara[j]=='tedong':
          if A[i-1]=='e':
              A[i-1]="o"
              i-=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  if A[i-3]=='e': 
                      A[i-3]="o"
                      i-=1
                  else :
                      A.append("g")
                      A[i-1]=A[i-2]
                      A[i-2]=A[i-3]
          elif A[i-1]=='r':
              if A[i-2]=='e':
                  A[i-2]="o"
                  i-=1
              else :
                  A.append("r")
                  A[i-1]=A[i-2]
          elif A[i-1]=='h':
              if A[i-2]=='e':
                  A[i-2]="o"
                  i-=1
              else :
                  A.append("h")
                  A[i-1]=A[i-2]
          else:
              A.append(A[i-1])
      elif aksara[j]=='ulu'or aksara[j]=='ulu_sari':
          A.append("ulu")
      elif aksara[j]=='ulu_ricem':
          A.append("ulu_ricem")
      elif aksara[j]=='ulu_candra':
          A.append("ulu_candra")
      elif aksara[j]=='pepet':
          A.append("pepet")
      elif aksara[j]=='surang':
          A.append("surang")
      elif aksara[j]=='cecek':
          A.append("cecek")
      elif aksara[j]=='suku' or aksara[j]=='suku_ilut':
          if A[i-1]=='r':
              A[i-2]="u"
              i-=1
          elif A[i-1]=='h':
              A[i-2]="u"
              i-=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A[i-3]="u"
                  i-=1
          elif A[i-1]=='a':
              A[i-1]="u"
              i-=1
          else :
              A.append("suku")
      elif aksara[j]=='bisah':
          A.append("h")
      elif aksara[j]=='adeg':
          A[i-1]=' '
          i-=1
      elif aksara[j]=='carik':
          if A[i-1]==' ':
              A[i-1]=","
              i-=1
          elif A[i-1]==',':
              A[i-1]="."
              i-=1
          else :
              A.append(",")
      
      elif aksara[j]=='carik_pamungkah':
          A.append(":")
      elif aksara[j]=='panten':
          A.append("   ")
      elif aksara[j]=='pemada':
          A.append("*akhir baris*")
          A.append(" ")
          i+=1
              
      #aksara wresastra        
      elif aksara[j]=='ha':
          if j==0:
              A.append("h")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="h"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="h"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="h"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="h"
              else :
                  A[i-1]="h"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="h"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="h"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          elif A[i-1]=='ulu_ricem':
              A.append("a")
              A.append("m")
              A[i-1]="h"
              i+=1
          elif A[i-1]=='ulu_candra':
              A.append("a")
              A.append("n")
              A.append("g")
              A[i-1]="h"
              i+=2
          else:
              A.append("h")
              A.append("a")
              i+=1
      
      
      elif aksara[j]=='na':
          if j==0:
              A.append("n")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="n"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="n"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="n"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="n"
              else :
                  A[i-1]="n"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="n"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="n"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("n")
              A.append("a")
              i+=1
              
              
      elif aksara[j]=='ca':
          if j==0:
              A.append("c")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="c"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="c"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="c"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="c"
              else :
                  A[i-1]="c"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="c"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="c"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("c")
              A.append("a")
              i+=1
      
      
      elif aksara[j]=='ra':
          if j==0:
              A.append("r")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="r"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="r"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="r"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="r"
              else :
                  A[i-1]="r"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="r"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="r"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("r")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ka':
          if j==0:
              A.append("k")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="k"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="k"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="k"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="k"
              else :
                  A[i-1]="k"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="k"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="k"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("k")
              A.append("a")
              i+=1
      
      elif aksara[j]=='da':
          if j==0:
              A.append("d")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="d"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="d"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="d"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="d"
              else :
                  A[i-1]="d"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="d"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="d"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("d")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ta':
          if j==0:
              A.append("t")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="t"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="t"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="t"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="t"
              else :
                  A[i-1]="t"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="t"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="t"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("t")
              A.append("a")
              i+=1
      
      elif aksara[j]=='sa':
          if j==0:
              A.append("s")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="s"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="s"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="s"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("s")
              A.append("a")
              i+=1
      
      elif aksara[j]=='wa':
          if j==0:
              A.append("w")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="w"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="w"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="w"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="w"
              else :
                  A[i-1]="w"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="w"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="w"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("w")
              A.append("a")
              i+=1
      
      elif aksara[j]=='la':
          if j==0:
              A.append("l")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="l"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="l"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="l"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="l"
              else :
                  A[i-1]="l"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="l"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="l"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("l")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ma':
          if j==0:
              A.append("n")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="m"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="m"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="m"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="m"
              else :
                  A[i-1]="m"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="m"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="m"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("m")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ga':
          if j==0:
              A.append("g")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="g"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="g"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="g"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="g"
              else :
                  A[i-1]="g"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="g"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="g"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("g")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ba':
          if j==0:
              A.append("b")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="b"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="b"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="b"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="b"
              else :
                  A[i-1]="b"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="b"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="b"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("b")
              A.append("a")
              i+=1
      
      
      elif aksara[j]=='nga':
          if j==0:
              A.append("n")
              A.append("g")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("g")
              A.append("e")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='ulu':
              A.append("g")
              A.append("i")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='pepet':
              A.append("g")
              A.append("q")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="n"
                  A[i-1]="g"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="n"
                  A.append("g")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="n"
                  A[i-1]="g"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="n"
                  A.append("g")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("n")
              A.append("g")
              A.append("a")
              i+=2
              
                  
      
      elif aksara[j]=='pa':
          if j==0:
              A.append("pa")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="p"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="p"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="p"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="p"
              else :
                  A[i-1]="p"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="p"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="p"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("p")
              A.append("a")
              i+=1
      
      
      elif aksara[j]=='ja':
          if j==0:
              A.append("j")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="j"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="j"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="j"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="j"
              else :
                  A[i-1]="j"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="j"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="j"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("j")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ya':
          if j==0:
              A.append("y")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="y"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="y"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="y"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="y"
              else :
                  A[i-1]="y"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="y"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="y"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("y")
              A.append("a")
              i+=1
      
              
      elif aksara[j]=='nya':
          if j==0:
              A.append("n")
              A.append("y")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("y")
              A.append("e")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='ulu':
              A.append("y")
              A.append("i")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='pepet':
              A.append("y")
              A.append("q")
              A[i-1]="n"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="n"
                  A[i-1]="y"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="n"
                  A.append("y")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="n"
                  A[i-1]="y"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="n"
                  A.append("y")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("n")
              A.append("y")
              A.append("a")
              i+=2
      
      #gantungan dan gempelan aksara wresastra
      elif aksara[j]=='gan_ha':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="h"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="h"
          else:
              A.append(A[i-1])
              A[i-1]="h"
              
      elif aksara[j]=='gan_na':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="n"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="n"
          else:
              A.append(A[i-1])
              A[i-1]="n"
              
      elif aksara[j]=='gan_ca':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="c"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="c"
          else:
              A.append(A[i-1])
              A[i-1]="c"
              
      elif aksara[j]=='gan_ra':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="r"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="r"
          else:
              A.append(A[i-1])
              A[i-1]="r"
              
      elif aksara[j]=='gan_ka':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="k"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="k"
          else:
              A.append(A[i-1])
              A[i-1]="k"
              
      elif aksara[j]=='gan_da':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="d"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="d"
          else:
              A.append(A[i-1])
              A[i-1]="d" 
              
      elif aksara[j]=='gan_ta':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="t"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="t"
          else:
              A.append(A[i-1])
              A[i-1]="t"
          
      elif aksara[j]=='gan_wa':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="w"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="w"
          else:
              A.append(A[i-1])
              A[i-1]="w"
              
      elif aksara[j]=='gan_la':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="l"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="l"
          else:
              A.append(A[i-1])
              A[i-1]="l"
              
      elif aksara[j]=='gan_ma':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="m"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="m"
          else:
              A.append(A[i-1])
              A[i-1]="m"
      
      elif aksara[j]=='gan_ga':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="g"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="g"
          else:
              A.append(A[i-1])
              A[i-1]="g"
              
      elif aksara[j]=='gan_ba':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="b"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="b"
          else:
              A.append(A[i-1])
              A[i-1]="b"
              
      elif aksara[j]=='gan_nga':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="n"
              A[i-1]="g"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="n"
                  A[i-2]="g"
                  i+=1
          else:
              A.append("g")
              A.append(A[i-1])
              A[i-1]="n"        
              
      elif aksara[j]=='gem_pa':
          if A[i-3]=='u':
              A[i-3]="s"
              if A[i-2]=='pepet':
                  A[i-2]="q"
                  if A[i-1]=='surang':
                      A[i-1]="r"
                      i-=1
                  elif A[i-1]=='cecek':
                      A[i-1]="n"
                      A.append("g")
              elif A[i-2]=='ulu':
                  A[i-2]="i"
                  if A[i-1]=='surang':
                      A[i-1]="r"
                      i-=1
                  elif A[i-1]=='cecek':
                      A[i-1]="n"
                      A.append("g")
              
          elif A[i-2]=='u':
              if A[i-1]=='ulu':
                  A[i-2]="s"
                  A[i-1]="i"
                  i-=1
              elif A[i-1]=='pepet':
                  A[i-2]="s"
                  A[i-1]="q"
                  i-=1
              elif A[i-1]=='surang':
                  A[i-2]="s"
                  A[i-1]="a"
                  A.append("r")
              elif A[i-1]=='cecek':
                  A[i-2]="s"
                  A[i-1]="a"
                  A.append("n")
                  A.append("g")
                  i+=1
          elif A[i-1]=='u':
              A[i-1]="s"
              A.append("a")
          elif A[i-2]=='suku':
              if A[i-1]=='surang':
                  A[i-1]="r"
                  A[i-2]=A[i-3]
                  A[i-3]="s"
                  i-=1
              elif A[i-1]=='cecek':
                  A.append("g")
                  A[i-1]="n"
                  A[i-2]=A[i-3]
                  A[i-3]="s"
          elif A[i-1]=='suku':
              A[i-1]=A[i-2]
              A[i-2]="s"
              i-=1
          elif A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="j"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="j"
      
          else:
              A.append(A[i-1])
              A[i-1]="p"        
              
      elif aksara[j]=='gan_ja':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="j"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="j"
          else:
              A.append(A[i-1])
              A[i-1]="j"
              
      elif aksara[j]=='gan_ya':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="i"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="i"
          else:
              A.append(A[i-1])
              A[i-1]="i"
      
      elif aksara[j]=='gan_nya':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="n"
              A[i-1]="y"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="n"
                  A[i-2]="y"
                  i+=1
          else:
              A.append("y")
              A.append(A[i-1])
              A[i-1]="n"
          
      #aksara suara
      elif aksara[j]=='a_kara':
          if j==0:
              A.append("-a")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-a"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-a"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-a"
          else :
              A.append("-a")
              
      elif aksara[j]=='i_kara':
          if j==0:
              A.append("-i")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-i"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-i"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-i"
          else :
              A.append("-i")
      
      elif aksara[j]=='u_kara':
          if j==0:
              A.append("-u")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-u"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-u"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-u"
          else :
              A.append("-u")
              
      elif aksara[j]=='e_kara':
          if j==0:
              A.append("-e")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-e"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-e"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-e"
          elif aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j-1]=='1' or aksara[j-1]=='la_lenga' or aksara[j-1]=='o_kara' or aksara[j-1]=='4' or aksara[j-1]=='5'or aksara[j-1]=='e_kara' or aksara[j-1]=='7' or aksara[j-1]=='pa_kapal' or aksara[j-1]=='9' or aksara[j-1]=='0' or aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j+1]=='1' or aksara[j+1]=='la_lenga' or aksara[j+1]=='o_kara' or aksara[j+1]=='4' or aksara[j+1]=='5'or aksara[j+1]=='6' or aksara[j+1]=='7' or aksara[j+1]=='pa_kapal' or aksara[j+1]=='9' or aksara[j+1]=='0':
              A.append("6")
          elif aksara[j+1]=='carik':
              A.append("6")
          else :
              A.append("-e")
      
      elif aksara[j]=='o_kara':
          if j==0:
              A.append("-o")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-o"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-o"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-o"
          elif aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j-1]=='1' or aksara[j-1]=='la_lenga' or aksara[j-1]=='o_kara' or aksara[j-1]=='4' or aksara[j-1]=='5'or aksara[j-1]=='e_kara' or aksara[j-1]=='7' or aksara[j-1]=='pa_kapal' or aksara[j-1]=='9' or aksara[j-1]=='0' or aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j+1]=='1' or aksara[j+1]=='la_lenga' or aksara[j+1]=='o_kara' or aksara[j+1]=='4' or aksara[j+1]=='5'or aksara[j+1]=='6' or aksara[j+1]=='7' or aksara[j+1]=='pa_kapal' or aksara[j+1]=='9' or aksara[j+1]=='0':
              A.append("3")
          elif aksara[j+1]=='carik':
              A.append("3")
          else :
              A.append("-o")
              
      elif aksara[j]=='ja_jera':
          if j==0:
              A.append("-ai")
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("n")
              A.append("g")
              A[i-1]="-ai"
              i+=1
          elif A[i-1]=='surang':
              A.append("r")
              A[i-1]="-ai"
          elif A[i-1]=='ulu_ricem':
              A.append("m")
              A[i-1]="-ai"
          else :
              A.append("-ai")
      #ra_repa & la_lenga
      elif aksara[j]=='ra_repa':
          if j==0:
              A.append("r")
              A.append("q")
              i+=1
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("q")
              A.append("n")
              A.append("g")
              A[i-1]="r"
              i+=2
          elif A[i-1]=='surang':
              A.append("q")
              A.append("r")
              A[i-1]="r"
              i+=1
          elif A[i-1]=='ulu_ricem':
              A.append("q")
              A.append("m")
              A[i-1]="r"
              i+=1
          else :
              A.append("r")
              A.append("q")
              i+=1
              
      elif aksara[j]=='la_lenga':
          if j==0:
              A.append("l")
              A.append("q")
              i+=1
          elif A[i-1]=='cecek'or A[i-1]=='ulu_candra':
              A.append("q")
              A.append("n")
              A.append("g")
              A[i-1]="l"
              i+=2
          elif A[i-1]=='surang':
              A.append("q")
              A.append("r")
              A[i-1]="l"
              i+=1
          elif A[i-1]=='ulu_ricem':
              A.append("q")
              A.append("m")
              A[i-1]="l"
              i+=1
          elif aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j-1]=='1' or aksara[j-1]=='la_lenga' or aksara[j-1]=='o_kara' or aksara[j-1]=='4' or aksara[j-1]=='5'or aksara[j-1]=='e_kara' or aksara[j-1]=='7' or aksara[j-1]=='pa_kapal' or aksara[j-1]=='9' or aksara[j-1]=='0' or aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j+1]=='1' or aksara[j+1]=='la_lenga' or aksara[j+1]=='o_kara' or aksara[j+1]=='4' or aksara[j+1]=='5'or aksara[j+1]=='6' or aksara[j+1]=='7' or aksara[j+1]=='pa_kapal' or aksara[j+1]=='9' or aksara[j+1]=='0':
              A.append("2")   
          elif aksara[j+1]=='carik':
              A.append("2")
          else :
              A.append("l")
              A.append("q")
              i+=1
              
      #gempelan ra_repa
      elif aksara[j]=='gem_ra_repa':
          A[i-1]=("r")
          A.append("q")
      
      elif aksara[j]=='guwung_mecelek':
          A[i-1]=("r")
          A.append("q")
      
      #angka bali
      elif aksara[j]=='1':
          A.append("1")
      elif aksara[j]=='4':
          A.append("4")
      elif aksara[j]=='5':
          A.append("5")
      elif aksara[j]=='7':
          A.append("7")
      elif aksara[j]=='9':
          A.append("9")
      elif aksara[j]=='0':
          A.append("0")
          
      #aksara wayah
      elif aksara[j]=='na_rambat':
          if j==0:
              A.append("n")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="n"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="n"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="n"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="n"
              else :
                  A[i-1]="n"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="n"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="n"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("n")
              A.append("a")
              i+=1
      
      elif aksara[j]=='da_madu':
          if j==0:
              A.append("d")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="d"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="d"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="d"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="d"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="d"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="d"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="d"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("d")
              A.append("h")
              A.append("a")
              i+=2   
      
      elif aksara[j]=='ta_latik':
          if j==0:
              A.append("t")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="t"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="t"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="t"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="t"
              else :
                  A[i-1]="t"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="t"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="t"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("t")
              A.append("a")
              i+=1
              
      elif aksara[j]=='ta_tawa':
          if j==0:
              A.append("t")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="t"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="t"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="t"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="t"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="t"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="t"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="t"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("t")
              A.append("h")
              A.append("a")
              i+=2   
      
      elif aksara[j]=='sa_sapa':
          if j==0:
              A.append("s")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="s"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="s"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="s"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("s")
              A.append("a")
              i+=1
      
      elif aksara[j]=='sa_saga':
          if j==0:
              A.append("s")
              A.append("a")
              i+=1
          elif A[i-1]=='taleng':
              A.append("e")
              A[i-1]="s"
          elif A[i-1]=='ulu':
              A.append("i")
              A[i-1]="s"
          elif A[i-1]=='pepet':
              A.append("q")
              A[i-1]="s"
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  A.append("r")
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("r")
                  i+=1
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A[i-1]="i"
                  elif A[i-2]=='pepet':
                      A[i-1]="q"
                  elif A[i-2]=='taleng':
                      A[i-1]="e"
                  A[i-2]="s"
                  A.append("n")
                  A.append("g")
                  i+=1
              else :
                  A[i-1]="s"
                  A.append("a")              
                  A.append("n")
                  A.append("g")
                  i+=2
          else:
              A.append("s")
              A.append("a")
              i+=1
      
      elif aksara[j]=='ga_gora':
          if j==0:
              A.append("g")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="g"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="g"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="g"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="g"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="g"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="g"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="g"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("g")
              A.append("h")
              A.append("a")
              i+=2 
      
      elif aksara[j]=='ba_kembang':
          if j==0:
              A.append("b")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="b"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="b"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="b"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="b"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="b"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="b"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="b"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("b")
              A.append("h")
              A.append("a")
              i+=2 
      
      elif aksara[j]=='pa_kapal':
          if j==0:
              A.append("p")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="p"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="p"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="p"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="p"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          elif aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j-1]=='1' or aksara[j-1]=='la_lenga' or aksara[j-1]=='o_kara' or aksara[j-1]=='4' or aksara[j-1]=='5'or aksara[j-1]=='e_kara' or aksara[j-1]=='7' or aksara[j-1]=='pa_kapal' or aksara[j-1]=='9' or aksara[j-1]=='0' or aksara[j-1]=='.' or aksara[j-1]==',' or aksara[j+1]=='1' or aksara[j+1]=='la_lenga' or aksara[j+1]=='o_kara' or aksara[j+1]=='4' or aksara[j+1]=='5'or aksara[j+1]=='6' or aksara[j+1]=='7' or aksara[j+1]=='pa_kapal' or aksara[j+1]=='9' or aksara[j+1]=='0':
              A.append("8")
          elif aksara[j+1]=='carik':
              A.append("8")
          else :
              A.append("p")
              A.append("h")
              A.append("a")
              i+=2 
              
      elif aksara[j]=='ka_mahaprana':
          if j==0:
              A.append("k")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="k"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="k"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="k"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="k"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="k"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="k"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="k"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("k")
              A.append("h")
              A.append("a")
              i+=2 
              
      elif aksara[j]=='ca_laca':
          if j==0:
              A.append("c")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="c"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="c"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="c"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="c"
                  A[i-1]="h"
                  A.append("r")
                  i+=1
              else :
                  A[i-1]="c"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                  A[i-2]="c"
                  A[i-1]="h"
                  A.append("n")
                  A.append("g")
                  i+=2
              else :
                  A[i-1]="c"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("c")
              A.append("h")
              A.append("a")
              i+=2         
              
      #gantungan dan gempelan aksara wayah
      elif aksara[j]=='gan_na_rambat':
          if A[i-1]=='r':
              A.append("r")
              A[i-1]=A[i-2]
              A[i-2]="n"
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append("g")
                  A[i-1]=A[i-2]
                  A[i-2]=A[i-3]
                  A[i-3]="n"
          else:
              A.append(A[i-1])
              A[i-1]="n"
      
      elif aksara[j]=='gan_da_madu':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="d"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="d"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="d"
              i+=1
              
      elif aksara[j]=='gan_ta_latik':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="t"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="t"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="t"
              i+=1
              
      elif aksara[j]=='gan_ta_tawa':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="t"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="t"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="t"
              i+=1
              
      elif aksara[j]=='gan_sa_saga':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="s"
              A[i-1]="y"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="s"
                  A[i-2]="y"
                  i+=1
          else:
              A.append("y")
              A.append(A[i-1])
              A[i-1]="s"
              i+=1
      
      elif aksara[j]=='gan_ga_gora':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="g"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="g"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="g"  
              i+=1
      
      elif aksara[j]=='gan_ba_kembang':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="b"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="b"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="b"  
              i+=1
      
      elif aksara[j]=='gan_ka_mahaprana':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="k"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="k"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="k"   
      
      elif aksara[j]=='gan_ca_laca':
          if A[i-1]=='r':
              A.append(A[i-2])
              A.append(A[i-1])
              A[i-2]="c"
              A[i-1]="h"
              i+=1
          elif A[i-1]=='g':
              if A[i-2]=='n':
                  A.append(A[i-2])
                  A.append(A[i-1])
                  A[i-1]=A[i-3]
                  A[i-3]="c"
                  A[i-2]="h"
                  i+=1
          else:
              A.append("h")
              A.append(A[i-1])
              A[i-1]="c"  
              i+=1
      
      elif aksara[j]=='gem_sa_sapa':
          if j==0:
              A.append("s")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="s"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="s"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="s"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                      A[i-2]="s"
                      A[i-1]="h"
                      A.append("r")
                      i+=1
              else :
                  A[i-1]="s"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                      A[i-2]="s"
                      A[i-1]="h"
                      A.append("n")
                      A.append("g")
                      i+=2
              else :
                  A[i-1]="s"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          else :
              A.append("s")
              A.append("h")
              A.append("a")
              i+=2  
                    
              
      elif aksara[j]=='gem_pa_kapal':
          if j==0:
              A.append("p")
              A.append("h")
              A.append("a")
              i+=2
          elif A[i-1]=='taleng':
              A.append("h")
              A.append("e")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='ulu':
              A.append("h")
              A.append("i")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='pepet':
              A.append("h")
              A.append("q")
              A[i-1]="p"
              i+=1
          elif A[i-1]=='surang':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                      A[i-2]="p"
                      A[i-1]="h"
                      A.append("r")
                      i+=1
              else :
                  A[i-1]="p"
                  A.append("h")
                  A.append("a")
                  A.append("r")
                  i+=2
          elif A[i-1]=='cecek':
              if A[i-2]=='ulu' or A[i-2]=='taleng' or A[i-2]=='pepet':
                  if A[i-2]=='ulu':
                      A.append("i")
                  elif A[i-2]=='pepet':
                      A.append("q")
                  elif A[i-2]=='taleng':
                      A.append("e");
                      A[i-2]="p"
                      A[i-1]="h"
                      A.append("n")
                      A.append("g")
                      i+=2
              else :
                  A[i-1]="p"
                  A.append("h")              
                  A.append("a")
                  A.append("n")
                  A.append("g")
                  i+=3
          
          else :
              A.append("p")
              A.append("h")
              A.append("a")
              i+=2   
      
              
      i+=1

  #print(''.join(A))

  z=0
  cetak = []
  for z in range(i):
      if A[z]=='q':
          cetak.append("e")
      elif A[z]=='e':
          cetak.append("e'")
      elif A[z]=='-a':
          cetak.append("a")
      elif A[z]=='-i':
          cetak.append("i")
      elif A[z]=='-u':
          cetak.append("u")
      elif A[z]=='-e':
          cetak.append("e")
      elif A[z]=='-o':
          cetak.append("o")
      elif A[z]=='-ai':
          cetak.append("ai")
      else :
          cetak.append(A[z])

  z=0
  result = ''
  for z in range(i):
      if cetak[z]=='*akhir baris*':
          result = result
      elif cetak[z]==',':
          result += ','
      else :
          result += cetak[z]
  
  print('Result:\n' + result)
  returnFix.config(text=result)
