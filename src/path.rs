use std::env;
use std::path::PathBuf;

pub fn get_current_dir() -> Result<String, Box<dyn std::error::Error>> {
    let path: PathBuf = env::current_dir()?;
    
    match path.into_os_string().into_string() {
        Ok(path_str) => Ok(path_str),
        Err(_) => Err("Path contains invalid UTF-8 characters".into())
    }
}