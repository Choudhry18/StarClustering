echo "downloading galaxy mosaics..."	
wget -P legus/frc_fits_files -q --show-progress -i frc_fits_links.txt

cd legus/frc_fits_files/
ls ./*.tar.gz |xargs -n1 tar -xvzf
rm -r ./*.tar.gz
cd ../../ 
echo "downloading tab files"	
wget -P legus/tab_files -q --show-progress -i tab_links.txt