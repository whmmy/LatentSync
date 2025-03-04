
cp process_script.sh /etc/init.d/process_script
update-rc.d process_script defaults
update-rc.d -n process_script defaults
service process_script start

chmod +x /etc/init.d/process_script
service process_script stop
service process_script start

cp ls_process.conf /etc/supervisor/conf.d/ls_process.conf
chmod +x /etc/supervisor/conf.d/ls_process.conf