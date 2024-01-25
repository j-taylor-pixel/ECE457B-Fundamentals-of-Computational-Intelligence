# open repo and 'nix-shell' to enter this shell
{ pkgs ? import <nixpkgs> {} }:
let
  my-python-packages = ps: with ps; [
    sklearn-deap
    # other python packages
  ];
  my-python = pkgs.python3.withPackages my-python-packages;
in my-python.env