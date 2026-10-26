# Diagnostic des hausses de volume entre époques (2026-10-26)

Une dissection ne fait qu'enlever du tissu : le volume d'une époque fermée ne devrait jamais dépasser celui de l'époque précédente. Sur les 104 pas entre époques consécutives (époques « b » exclues), **15 montrent une hausse** (+0,3 à +15,3 %). Chaque hausse a été comparée à 7 paires « normales » servant de témoins.

Scripts : `abstract/scripts/diagnose_volume_increases.py`, `cap_plane_offset.py`, `classify_increases.py`. Chiffres : `volume_increases.csv`, `cap_plane_offset.txt`.

## Ce qui a été écarté

| Hypothèse | Mesure | Résultat |
|---|---|---|
| Unités / repère différent | bornes, ICP | Tout est en mètres. Les époques d'un spécimen sont déjà dans le même repère : la distance médiane d'une époque à la précédente est de 0,4 à 0,7 mm, et la translation ICP de 0,07 à 0,36 mm. |
| Échelle | ICP de similarité tronqué (60 % des paires les plus proches), étendue le long du grand axe | Facteur de 0,992 à 1,008 pour les hausses, contre 0,998 à 1,007 pour les témoins : c'est indiscernable du bruit. Le grand axe reste stable à ±0,5 mm dans la plupart des cas. L'échelle ne peut expliquer au plus que ~±2 % de volume, et rien ne la distingue des témoins. |
| Coin de volume ajouté par les caps | cap comparé à un éventail sur les mêmes boucles | 0,04 à 2,0 % par époque, comme chez les témoins. Ne peut pas expliquer +13 à +15 %. |

## Causes trouvées

| Cause | Paires | Chiffres |
|---|---|---|
| **Face de coupe déplacée** : le cap principal et sa boucle (l'endroit où la surface scannée s'arrête) sont plus à l'extérieur à l'époque suivante | 9 / 15 : spc-07 09→10 (+13,4 %), spc-12 02→03, spc-15 10→11 (+5,8 %), spc-15 13→14, spc-17 02→03 (+6,8 %), spc-17 07→08 (+15,3 %), spc-18 02→03, spc-19 08→09, spc-19 11→12 (+12,9 %) | Cap déplacé de +2,2 à +5,1 mm vers l'extérieur ; la boucle bouge d'autant (écart < 0,6 mm entre les deux), avec des plans quasi parallèles (< 1,5°). Tranche correspondante : 19 à 57 cm³, du même ordre que les hausses (p. ex. spc-17 07→08 : +34 cm³ de volume pour une tranche de 40 cm³). 73 à 93 % du volume de l'époque suivante situé hors de la précédente est à moins de 5 mm d'un cap (témoins : 8 à 25 %, sauf spc-01 01→02 à 79 % et spc-18 03→04 à 69 %). L'épaisseur de l'hémisphère (3ᵉ axe principal) augmente de 3 à 6 mm pour les plus grosses hausses (spc-07 09→10, spc-17 02→03 et 07→08, spc-18 02→03, spc-19 11→12) ; pour les autres, elle varie de −0,2 à +2,7 mm. |
| **Fermeture de l'époque précédente** (ouvertures repliées, caps courbes) | 2 / 15 : spc-07 03→05 (+3,2 %), spc-15 08→09 (+5,2 %) | L'époque précédente a 9 et 4 caps courbes et 54 et 41 ouvertures, contre 1 et 0 caps courbes et 5 et 1 ouvertures pour l'époque suivante. spc-15 epo-08 est l'époque des « trous sous le cap » (fig. 3b). Le cap principal ne bouge pas (+1,3 et 0,0 mm). |
| **Dans l'incertitude** (< 3,4 %) | 4 / 15 : spc-02 07→08, spc-10 05→06, spc-12 05→06, spc-16 06→07 (+0,3 à +1,9 %) | Cap déplacé de 0,5 à 1,2 mm, ce qui est dans la plage des témoins. |

## Conclusion

- **Les grosses hausses viennent du maillage, pas de la fermeture.** La surface scannée ne s'arrête pas au même endroit d'une époque à l'autre près de la face non photographiée (face médiale/de coupe). Le cap suit fidèlement ce bord, et toute fermeture raisonnable (plane, en éventail, au plan moyen) donnerait le même décalage, puisque c'est la boucle elle-même qui bouge.
- **Ordre de grandeur utile pour l'abstract** : chez les témoins, le cap bouge déjà de −2,2 à +1,9 mm d'une époque à l'autre. Avec une face de coupe d'environ 70 à 120 cm², 1 mm vaut 7 à 12 cm³, soit 2 à 4 % du volume. **L'incertitude sur la position de la face de coupe (~±2 mm, soit ~±5 à 8 %) dépasse celle de la fermeture (≤ 3,4 %)**, et elle est du même ordre que le volume retiré à une étape typique (médiane : 8,7 %).
- Deux hausses relèvent en partie de nos fermetures imparfaites (ouvertures repliées de spc-07 et spc-15 epo-08).

## Réserves (non vérifié)

- Je ne sais pas *pourquoi* le bord scanné bouge : affaissement du tissu sur son support, recadrage manuel différent des maillages (support, table), couverture photographique différente, ou recalage des époques entre elles par BraDiPho. Il faudrait la documentation BraDiPho, ou les photos originales.
- Je n'ai pas vérifié qu'aucun tissu n'est retiré du côté médial pendant la dissection. Si c'était le cas, une partie des déplacements *négatifs* du cap serait réelle.
- L'ICP de similarité est perturbé par le tissu retiré entre les époques. Le facteur d'échelle n'est fiable qu'à ~±0,5 % près.
- Les voxels font 1,5 mm : le volume « hors de l'autre époque » contient environ 4 à 10 cm³ de bruit de discrétisation (valeur des témoins).
- Les seuils de classification (2 mm, 70 %) ont été choisis à partir de 7 témoins seulement.
