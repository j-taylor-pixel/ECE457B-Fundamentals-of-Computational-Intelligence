{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.matplotlib
    python311Packages.numpy
    zlib
  ]);
  runScript = "bash";
}).env

# see: https://nixos.wiki/wiki/Python
# for direction on how to enter and use the environment in nixos
# tldr:
# nix-shell pip-shell.nix
# virtualenv venv
# source venv/bin/activate
# pip install -r requirements.txt