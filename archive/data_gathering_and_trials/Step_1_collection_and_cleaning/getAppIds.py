# This programs the the list of all the appids present on steam
import requests
import json

def get_sorted_steam_app_ids():
    # API endpoint for Steam's app list
    url = "https://api.steampowered.com/ISteamApps/GetAppList/v2?json=1"
    
    try:
        # Make the API request
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract the app list
            apps = data.get('applist', {}).get('apps', [])
            
            # Get the total count of appids
            total_count = len(apps)
            
            # Sort the apps by appid in ascending order
            sorted_apps = sorted(apps, key=lambda x: x['appid'])
            
            # Create the same structure but with sorted apps
            sorted_data = {"applist": {"apps": sorted_apps}}
            
            # Save to a JSON file
            with open('../appList/sorted_steam_apps.json', 'w') as f:
                json.dump(sorted_data, f, indent=4)
            
            print(f"Total number of appids: {total_count}")
            print(f"Successfully saved {total_count} sorted apps to sorted_steam_apps.json")
            return True
            
        else:
            print(f"Error: Received status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    get_sorted_steam_app_ids()