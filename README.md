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

## installing ensembl-vep
```bash
conda create -n vep
conda activate vep
conda install -y -c conda-forge -c bioconda -c defaults ensembl-vep==111.0 htslib==1.17 bcftools==1.17 samtools==1.18 ucsc-liftover==377
```

Install offline cache (database)
```bash
vep_install -a cf -s homo_sapiens -y GRCh38 -c $HOME/.vep --CONVERT
```

Or if it does not work correctly, download manually
```bash
mkdir -p $HOME/.vep/homo_sapiens/111_GRCh38/
rsync -avr --progress rsync://ftp.ensembl.org/pub/release-111/variation/indexed_vep_cache/homo_sapiens_vep_111_GRCh38.tar.gz $HOME/.vep/
tar -zxf $HOME/.vep/homo_sapiens_vep_111_GRCh38.tar.gz -C $HOME/.vep/
rsync -avr --progress rsync://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna_index/ $HOME/.vep/homo_sapiens/111_GRCh38/
```

Test the vep installation
```bash
curl -sLO https://raw.githubusercontent.com/Ensembl/ensembl-vep/release/111/examples/homo_sapiens_GRCh38.vcf
vep --species homo_sapiens --assembly GRCh38 --offline --no_progress --no_stats --sift b --ccds --uniprot --hgvs --symbol --numbers --domains --gene_phenotype --canonical --protein --biotype --tsl --pubmed --variant_class --shift_hgvs 1 --check_existing --total_length --allele_number --no_escape --xref_refseq --failed 1 --vcf --minimal --flag_pick_allele --pick_order canonical,tsl,biotype,rank,ccds,length --dir $HOME/.vep --cache --input_file homo_sapiens_GRCh38.vcf --output_file homo_sapiens_GRCh38.vep.vcf --polyphen b --af --af_1kg --af_esp --regulatory #--fasta $HOME/.vep/homo_sapiens/111_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz
```

source : https://stackoverflow.com/questions/70801077/how-to-run-ensembl-vep-in-conda
