## TP 4 #------------- EXERCICE 1 ------------

#### Partie A ####
p=5
n=50
h=matrix(nrow=p,ncol=n)

for (i in 1:p){
  h[i,]= rnorm(50, mean = c(0,0,0,0,0), sd = c(1,1,1,1,1))
}

S=cov(t(h))
View(S)

# REMARQUES
# termes diagonaux : proches vers 1 
# autres termes : proches de 0 (normal car les Xi sont iid)


# Decomposition en Valeurs Singulieres
SVD=svd(S)
U=SVD$u
V=SVD$v
Sigma=diag(SVD$d)

# Sigma contient les valeurs singulières de la matrice de covariance
# Deux méthodes pour comparer U et V :

sum(U!=V)
# norme L2
norm(U-V,type=("2"))
# on compare la norme L1
norm(V-U,type=("1"))

# les 2 matrices sont égales à la précision e-16

 
sum(diag(Sigma)) 
sum(diag(S))

# La décomposition en valeurs singulières conserve la trace, 
# comme la trace est invariante par changement de base.

# S est symétrique donc le theorème spectral s'applique : S = UDU* avec U*=Ut (orthogonale).
# Les valeurs singulières sont donc les valeurs propres et la trace est ainsi conservée.

# Vérification :
S2=U%*%Sigma%*%t(V)
norm(S2-S) 
# 2.276825e-15 qui est presque du même ordre que la précision de la représentation à virgule flottante.


#------------- EXERCICE 2 : ACP ------------

##### ANALYSE PRELIMINAIRE #####


data = read.table("cardata.txt", header = T, sep = ";", row.names = 1)
# 1,2 et 3 :
# Il y a 6 variables, 24 obs.
# On remarque que les unites des variables sont differentes. Il faut les normaliser.


plot(data) # Graphique = corrplot des variables
# Ex : On remarque que la puissance et la vitessse sont positivement corrélées et de même tous les plot qui ressemblent à une droite.

cor(data)
# Vit et Long, poids et cylind très fortement corrélees. 
# Les nombreuses corr?lations peuvent laisser penser que l'ACP
# va nous permettre de dégager des composantes principales pertinentes.



##### ACP #####

res = prcomp(data,scale.=T)
attributes(res) 
res$sdev # racine des valeurs propres de la mat de cov.
res$rotation # matrice des vecteurs propres de la matrice de cov
res$center # données pour centrer 'data' (moyennes)
res$scale # données pour normaliser 'data'(variances)
res$x # donne la matrice à l'échelle après projection sur vecteurs propres.


summary(res)

# On lit la proportion de variance expliquée par chaque axe principale sur la deuxième ligne.
# La proportion cumulée est sur la troisième ligne.
# Pour expliquer 95% de la variance on prend trois composantes.
# 4 composantes pour en expliquer 98 %.
# La variance totale est la somme des variances car indépendance après ACP

# remarques 
# CP 1 : chaque valeur positivement corrélée à hauteur d'environ ~ 0.4
# CP 2 : cylind, puiss, vit corrélée positivement. poid, long et larg le sont négativement.
# CP 3 : vit, long et larg are corr+ ;  cylind, puiss et poids corr-
# CP 4 : cylind puiss, poid et larg corr+ ; vit et long corr-

S = cov(res$x)
SVD = svd(S)
plot(SVD$d)

vpropres = res$sdev^2
var = sum(vpropres)

# La variance totale est la somme des variances car indépendance après ACP
# On trouve var = 6.

pourcentage_variance = vpropres/var
# On retrouve le résultat donné par summary(res)
# sur le pourcentage de variance expliquée


#### ETUDE DES VECTEURS PROPRES #####

res$rotation

#Les deux premiers axes ont v?ritablement un sens. Les axes 3 et 4 sont plus difficiles ? interpr?ter.
#Le premier axe est le classique "size effect".
#Le deuxi?me axe s?pare les voitures sportives des voitures moins sportives.
#C'est plus difficile ? interpr?ter ensuite.


##### ANALYSE DES INDIVIDUS PROJETES ET CERCLE DES CORRELATIONS #####

library(ade4)

res2=dudi.pca(data,scannf=F,nf=6)
biplot(res,choices=1:2)
s.corcircle(res2$co[,1:2])


# Projection des individus sur les plans factoriels 1 et 2.
# Les points sont très dispersés, l'ACP traduit bien la variance des données.
# Le premier axe indique de grands coefficients dans de nombreuses caractéristiques.
# Le deuxième axe oppose plutôt le moteur aux autres caractéristiques.
# (d'ou la peugeot 205 au dessus)

# Si un point est sur le diamètre du cercle, cela signifie que l'information est complètement captée par les deux composantes principales.
# Si un point est proche du centre, c'est le contraire.
# On retrouve par exemple que vitesse et largeur sont faiblement corrélées.

biplot(res,choices=2:3)
s.corcircle(res2$co[,2:3])


#### Classification Non Supervisee (kmeans) #####


resk1 = kmeans(data,1)
resk1
resk1$centers # coordonnées des centres des clusters
resk1$cluster # vecteur qui contient les obs associées à chaque cluster
resk1$withinss # vecteur de taille k qui contient la somme des distances au carré (intra cluster)
resk1$betweenss # même chose pour les points qui ne sont pas dans le même cluster

resk2 = kmeans(data,2)
resk2
resk2$centers
resk2$cluster
resk2$withinss
resk2$betweenss

resk3 = kmeans(data,3)
resk3
resk3$centers
resk3$cluster
resk3$withinss
resk3$betweenss

resk4 = kmeans(data,4)
resk4
resk4$centers
resk4$cluster
resk4$withinss
resk4$betweenss


# k = 1 : ne fait pas sens pour la classification (car on veut que les voitures soient "différentes")
# k = 2 : un cluster avec les voitures puissantes/lourdes, un avec les plus légères.
# k = 3 : un cluster avec une classe "moyenne" en plus.
# k = 4 : peu d'info en plus, juste une nouvelle classe moyenne.




# Plot PCA and clusters
plot(res$x[,1:2],col=resk4$cluster)
text(res$x[,1:2],rownames(res$x))

# On distingue clairement deux classes à droite du graphique, 
# les deux autres correspondant aux voitures lourdes sont plus imbriquées. 

# La meilleur classification semble être celle avec 3 clusters :
plot(res$x[,1:2],col=resk3$cluster)
text(res$x[,1:2],rownames(res$x))




#------------- EXERCICE 3  ------------

##### Partie A #####

load("digits3-8.RData")

mImage = function(vect)
{
  image(t(matrix(vect,16,16)), axes=FALSE, col = gray(0:255/255))
}


##### Partie B ######
mImage(d3[2,])
mImage(d3[3,])
mImage(d8[1,])
mImage(d8[2,])
mImage(d8[100,])
d3train=d3[sample(1:1100,1000),]
d3test=d3[sample(1:1100,100),]
d8train=d8[sample(1:1100,1000),]
d8test=d8[sample(1:1100,100),]


mImage(colSums(d3train)) # On dechiffre un 3
mImage(colSums(d8train)) # On déchiffre (à peu près) un 8


d3train = t(scale(t(d3train)))
d8train = scale(d8train)
d3train_cov = cov(d3train)
d8train_cov = cov(d8train)

s3 = svd(d3train)
sigma3 = s3$d
eigen3 = eigen(d3train_cov)
max(abs(eigen3$values - sigma3^2/1000))

# La matrice de covariance s'écrit (1/n)*d3train.t_d3train
# On decompose d3train en valeurs singulieres : d3train = U Sigma t(V)
# Donc 1/n d3train t_d3train) = 1/n V.Sigma^2.t_V
# Les valeurs propres de la matrice de covariance sont 1/n * (val. singulieres d3train)^2

s8 = svd(d8train)
sigma8 = s8$d
eigen8 = eigen(d8train_cov)

# modes propres pour d3 :
mImage(eigen3$vectors[,1]%*%t(eigen3$vectors[,1]))
mImage(eigen3$vectors[,2]%*%t(eigen3$vectors[,2]))
mImage(eigen3$vectors[,100]%*%t(eigen3$vectors[,100]))
mImage(eigen3$vectors[,200]%*%t(eigen3$vectors[,200]))

# on peut deviner la forme du 3 dans les premiers elements,
# puis ca s'efface au fur et a mesure que l'importance du mode propre decroit

# modes propres pour d8 :
mImage(eigen8$vectors[,1]%*%t(eigen8$vectors[,1]))
mImage(eigen8$vectors[,2]%*%t(eigen8$vectors[,2]))
mImage(eigen8$vectors[,100]%*%t(eigen8$vectors[,100]))
mImage(eigen8$vectors[,200]%*%t(eigen8$vectors[,200]))


# Mêmes remarques
proj3 = rbind(t(eigen3$vectors[,1:5]),matrix(0,nr = 251, ncol = 256));
max(abs(proj3%*%proj3 - proj3))

proj8 = rbind(t(eigen8$vectors[,1:5]),matrix(0,nr = 251, ncol = 256));
max(abs(proj8%*%proj8 - proj8))


# Si on reconstruit l'image en ne gardant que certains vecteurs principaux, on effecue une sorte de compression d'image.
# En ne gardant que les 10 premières valeurs singulières, on explique une bonne proportion de la variance.
# On pourra retrouver approximativement les coordonnees initiales grâce à la transformation inverse, et en terme de mémoire on a un gain important.
# (2560 valeurs et pas 256^2). 

