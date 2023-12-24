#!/bin/bash
echo ""
echo -e "Bypass MDM"
echo ""
echo -e "Bypass on Recovery"
if [ -d "/Volumes/Macintosh HD - Data" ]; then
  diskutil rename "Macintosh HD - Data" "Data"
fi
echo -e "User default: macbook, password: 1234"
realName="${realName:=Macbook}"
username="${username:=macbook}"
passw="${passw:=1234}"
dscl_path='/Volumes/Data/private/var/db/dslocal/nodes/Default' 
echo -e "Đang tạo user"
# Create user
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" UserShell "/bin/zsh"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" RealName "$realName"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" RealName "$realName"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" UniqueID "501"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" PrimaryGroupID "20"
mkdir "/Volumes/Data/Users/$username"
dscl -f "$dscl_path" localhost -create "/Local/Default/Users/$username" NFSHomeDirectory "/Users/$username"
dscl -f "$dscl_path" localhost -passwd "/Local/Default/Users/$username" "$passw"
dscl -f "$dscl_path" localhost -append "/Local/Default/Groups/admin" GroupMembership $username
echo "0.0.0.0 deviceenrollment.apple.com" >>/Volumes/Untitled/etc/hosts
echo "0.0.0.0 mdmenrollment.apple.com" >>/Volumes/Untitled/etc/hosts
echo "0.0.0.0 iprofiles.apple.com" >>/Volumes/Untitled/etc/hosts
echo -e "Chặn host thành công"
# echo "Remove config profile"
touch /Volumes/Data/private/var/db/.AppleSetupDone
csrutil disable
rm -rf /Volumes/Untitled/var/db/ConfigurationProfiles/Settings/.cloudConfigHasActivationRecord
rm -rf /Volumes/Untitled/var/db/ConfigurationProfiles/Settings/.cloudConfigRecordFound
touch /Volumes/Untitled/var/db/ConfigurationProfiles/Settings/.cloudConfigProfileInstalled
touch /Volumes/Untitled/var/db/ConfigurationProfiles/Settings/.cloudConfigRecordNotFound
csrutil enable
echo -e "Remove config profile thành công"
echo "----------------------"
