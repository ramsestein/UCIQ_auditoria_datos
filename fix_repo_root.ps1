# Fix repository root: remove global .git and init repo in project folder
if (Test-Path 'C:\.git') {
    Remove-Item -LiteralPath 'C:\.git' -Recurse -Force
    Write-Output 'Removed C:\.git'
} else {
    Write-Output 'No .git at C:\'
}

Set-Location 'C:\Users\Ramsés\Desktop\Proyectos\wave_studies'
if (-not (Test-Path '.git')) {
    git init
    Write-Output 'Initialized git repo'
} else {
    Write-Output '.git already exists in project folder'
}

# Reset remote origin if present
try { git remote remove origin } catch { }
try { git remote add origin https://github.com/ramsestein/UCIQ_auditoria_datos } catch { }

# Add and commit
git add -A
try {
    git commit -m 'Reorganize project: move auditory scripts, add READMEs, add .gitignore'
} catch {
    Write-Output 'No changes to commit or commit failed'
}

# Ensure main branch and push
try { git branch -M main } catch { }
try { git push -u origin main } catch { Write-Output 'Push may have failed — check credentials or remote.' }
Write-Output 'Done.'
