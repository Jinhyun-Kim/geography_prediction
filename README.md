# Setting up the environment

## installing python environment
```bash
conda env create -f environment.yml -n py39
```

## installing ANNOVAR db
```bash
ANNOVAR_PATH="/home/jinhyun/tools/annovar"
REF_BUILD_VER="hg38"

# gene annotation
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar refgene $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar knownGene $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar ensGene $ANNOVAR_PATH/humandb/ 

# VEP
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar dbnsfp42c $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar intervar_20180118 $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar avsnp150 $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar clinvar_20221231 $ANNOVAR_PATH/humandb/ 

## VAF data
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar gnomad40_genome $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar 1000g2015aug $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar esp6500siv2_all $ANNOVAR_PATH/humandb/ 
$ANNOVAR_PATH/annotate_variation.pl -buildver $REF_BUILD_VER -downdb -webfrom annovar exac03 $ANNOVAR_PATH/humandb/ 

#$ANNOVAR_PATH/table_annovar.pl $INPUT.avinput $ANNOVAR_PATH/humandb/ -buildver hg19 -out $INPUT -remove -protocol refGene,1000g2015aug_all,avsnp150,cosmic70,icgc28,dbnsfp42c -operation g,f,f,f,f,f -nastring . 

```