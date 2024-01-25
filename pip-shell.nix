{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.matplotlib
  ]);
  runScript = "bash";
}).env

# see: https://nixos.wiki/wiki/Python
# for direction on how to enter and use the environment in nixos