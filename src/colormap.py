# https://rdrr.io/github/GreenleafLab/ArchR/src/R/ColorPalettes.R

# https://matplotlib.org/stable/api/cm_api.html
# https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#20-colors
stallion = ["#D51F26","#272E6A","#208A42","#89288F","#F47D2B", "#FEE500","#8A9FD1","#C06CAB","#E6C2DC",
            "#90D5E4", "#89C75F","#F37B7D","#9983BD","#D24B27","#3BBCA8", "#6E4B9E","#0C727C", "#7E1416","#D8A767","#3D3D3D"]

stallion2 = ["#D51F26","#272E6A","#208A42","#89288F","#F47D2B", "#FEE500","#8A9FD1","#C06CAB","#E6C2DC",
            "#90D5E4", "#89C75F","#F37B7D","#9983BD","#D24B27","#3BBCA8", "#6E4B9E","#0C727C", "#7E1416","#D8A767"]

calm = ["#7DD06F", "#844081", "#688EC1", "#C17E73", "#484125", "#6CD3A7", "#597873","#7B6FD0", "#CF4A31", "#D0CD47",
        "#722A2D", "#CBC594", "#D19EC4", "#5A7E36", "#D4477D", "#403552", "#76D73C", "#96CED5", "#CE54D1", "#C48736"]

kelly = ["#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020", "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A",
        "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800", "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16"]

#16-colors
bear = ["#faa818", "#41a30d","#fbdf72", "#367d7d",  "#d33502", "#6ebcbc", "#37526d",
        "#916848", "#f5b390", "#342739", "#bed678","#a6d9ee", "#0d74b6",
        "#60824f","#725ca5", "#e0598b"]

#15-colors
ironMan = ['#371377','#7700FF','#9E0142','#FF0080', '#DC494C',"#F88D51","#FAD510","#FFFF5F",'#88CFA4',
        '#238B45',"#02401B", "#0AD7D3","#046C9A", "#A2A475", '#595959']
ironMan_2 = ['#7700FF','#9E0142','#FF0080', '#DC494C',"#F88D51","#FAD510","#FFFF5F",'#88CFA4',
        '#238B45',"#02401B", "#0AD7D3","#046C9A", "#A2A475", '#595959']

circus = ["#D52126", "#88CCEE", "#FEE52C", "#117733", "#CC61B0", "#99C945", "#2F8AC4", "#332288",
            "#E68316", "#661101", "#F97B72", "#DDCC77", "#11A579", "#89288F", "#E73F74"]

#12-colors
paired = ["#A6CDE2","#1E78B4","#74C476","#34A047","#F59899","#E11E26",
            "#FCBF6E","#F47E1F","#CAB2D6","#6A3E98","#FAF39B","#B15928"]

#11-colors
grove = ["#1a1334","#01545a","#017351","#03c383","#aad962","#fbbf45","#ef6a32","#ed0345","#a12a5e","#710162","#3B9AB2"]
grove2 = ["#08A8CE","#017351",'#56A65A',"#03c383","#aad962","#fbbf45","#ef6a32","#ed0345","#a12a5e","#710162","#3B9AB2"]

#7-colors
summerNight = ["#2a7185", "#a64027", "#fbdf72","#60824f","#9cdff0","#022336","#725ca5"]

#5-colors
zissou = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"] #wesanderson
zissou2 = ["#007EB7","#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"] 
darjeeling = ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"] #wesanderson
rushmore = ["#E1BD6D", "#EABE94", "#0B775E", "#35274A" , "#F2300F"] #wesanderson
captain = ["grey","#A1CDE1","#12477C","#EC9274","#67001E"]

#---------------------------------------------------------------
# Primarily Continuous Palettes
#---------------------------------------------------------------

#10-colors
horizon = ['#000075','#2E00FF', '#9408F7', '#C729D6', '#FA4AB5', '#FF6A95', '#FF8B74', '#FFAC53', '#FFCD32', '#FFFF60']

#9-colors
horizonExtra =["#000436","#021EA9","#1632FB","#6E34FC","#C732D5","#FD619D","#FF9965","#FFD32B","#FFFC5A"]
blueYellow = ["#352A86","#343DAE","#0262E0","#1389D2","#2DB7A3","#A5BE6A","#F8BA43","#F6DA23","#F8FA0D"]
sambaNight = ['#1873CC','#1798E5','#00BFFF','#4AC596','#00CC00','#A2E700','#FFFF00','#FFD200','#FFA500'] #buencolors
solarExtra = ['#3361A5', '#248AF3', '#14B3FF', '#88CEEF', '#C1D5DC', '#EAD397', '#FDB31A', '#E42A2A', '#A31D1D']  #buencolors
whitePurple = ['#f7fcfd','#e0ecf4','#bfd3e6','#9ebcda','#8c96c6','#8c6bb1','#88419d','#810f7c','#4d004b']
whiteBlue = ['#fff7fb','#ece7f2','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#045a8d','#023858']
whiteRed = ["white", "red"]
comet = ["#E6E7E8","#3A97FF","#8816A7","black"]

#7-colors
greenBlue = ['#e0f3db','#ccebc5','#a8ddb5','#4eb3d3','#2b8cbe','#0868ac','#084081']

#6-colors
beach = ["#87D2DB","#5BB1CB","#4F66AF","#F15F30","#F7962E","#FCEE2B"]

#5-colors
coolwarm = ["#4858A7", "#788FC8", "#D6DAE1", "#F49B7C", "#B51F29"]
fireworks = ["white","#2488F0","#7F3F98","#E22929","#FCB31A"]
greyMagma = ["grey", "#FB8861FF", "#B63679FF", "#51127CFF", "#000004FF"]
fireworks2 = ["black", "#2488F0","#7F3F98","#E22929","#FCB31A"]
fireworks3 = ["#2488F0","#7F3F98","#E22929","#FCB31A"]

purpleOrange = ["#581845", "#900C3F", "#C70039", "#FF5744", "#FFC30F"]


##### backup color map
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# cmap1 = LinearSegmentedColormap.from_list("mycmap", zissou2)
# color_map1=cmap1
# cmap1 = LinearSegmentedColormap.from_list("mycmap", beach)
# color_map2=cmap1
# cmap1 = LinearSegmentedColormap.from_list("mycmap", purpleOrange)
# color_map3=cmap1
# cmap1 = LinearSegmentedColormap.from_list("mycmap", fireworks3)
# color_map4=cmap1






# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# # %%
# cmap1 = LinearSegmentedColormap.from_list("mycmap", blueYellow)
# color_map=cmap1
# plt.scatter(embedding[:,0],embedding[:,1],s=1,c=one_gene.alpha,cmap=color_map)


# # %%
# cmap1 = LinearSegmentedColormap.from_list("mycmap", fireworks2)
# color_map=cmap1
# for i in gene_choice:
#     save_path="output/detailcsv/adj_e/"+i+"_"+"e"+str(e_num)+".pdf"# notice: changed
#     velocity_plot(detail, [i],detailfinfo,color_scatter,20,1,color_map,vmin,vmax,save_path,step_i,step_j) # from cell dancer
#map_list=[stallion,stallion2,calm,kelly,bear,ironMan,ironMan_2,circus,paired,grove,summerNight]
