#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
import chardet
import random
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
import nltk
import javalang
import jieba



def get_ast(code):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parse.Parser(tokens)
    ast = parser.parse_member_declaration()
    return ast


def get_structure(ast):
    global code_structure
    ast_type = type(ast)
    if ast_type == javalang.tree.ClassDeclaration:
        code_structure += 'ClassDeclaration '
        if ast.body is not None:
            for t in ast.body:
                get_structure(t)
    elif ast_type == javalang.tree.MethodDeclaration:
        code_structure += 'MethodDeclaration '
        if ast.body is not None:
            for t in ast.body:
                get_structure(t)
    elif ast_type == javalang.tree.ConstructorDeclaration:
        code_structure += 'ConstructorDeclaration '
        if ast.body is not None:
            for t in ast.body:
                get_structure(t)
    elif ast_type == javalang.tree.InterfaceDeclaration:
        code_structure += 'InterfaceDeclaration '
        if ast.body is not None:
            for t in ast.body:
                get_structure(t)
    elif ast_type == javalang.tree.AnnotationDeclaration:
        code_structure += 'AnnotationDeclaration '
        if ast.body is not None:
            for t in ast.body:
                get_structure(t)
    elif ast_type == javalang.tree.SynchronizedStatement:
        code_structure += 'SynchronizedStatement '
        if ast.block is not None:
            for t in ast.block:
                get_structure(t)
    elif ast_type == javalang.tree.LocalVariableDeclaration:
        code_structure += 'LocalVariableDeclaration '
    elif ast_type == javalang.tree.ForStatement:
        code_structure += 'ForStatement '
        if ast.body is not None:
            get_structure(ast.body)
    elif ast_type == javalang.tree.BlockStatement:
        code_structure += 'BlockStatement '
        if ast.statements is not None:
            for t in ast.statements:
                get_structure(t)
    elif ast_type == javalang.tree.IfStatement:
        code_structure += 'IfStatement '
        if ast.then_statement is not None:
            get_structure(ast.then_statement)
        if ast.else_statement is not None:
            get_structure(ast.else_statement)
    elif ast_type == javalang.tree.TryStatement:
        code_structure += 'TryStatement '
        if ast.block is not None:
            for t in ast.block:
                get_structure(t)
    elif ast_type == javalang.tree.WhileStatement:
        code_structure += 'WhileStatement '
        if ast.body is not None:
            get_structure(ast.body)
    elif ast_type == javalang.tree.DoStatement:
        code_structure += 'DoStatement '
        if ast.body is not None:
            get_structure(ast.body)
    elif ast_type == javalang.tree.StatementExpression:
        code_structure += 'StatementExpression '
    elif ast_type == javalang.tree.ReturnStatement:
        code_structure += 'ReturnStatement '
    elif ast_type == javalang.tree.ConstantDeclaration:
        code_structure += 'ConstantDeclaration '
    elif ast_type == javalang.tree.FieldDeclaration:
        # code_structure += 'FieldDeclaration '
        if ast.declarators:
            for t in ast.declarators:
                get_structure(t)
    elif ast_type == javalang.tree.EnumDeclaration:
        code_structure += 'EnumDeclaration '
    elif ast_type == javalang.tree.ThrowStatement:
        code_structure += 'ThrowStatement '
    elif ast_type == javalang.tree.SwitchStatement:
        code_structure += 'SwitchStatement '
    elif ast_type == javalang.tree.AssertStatement:
        code_structure += 'AssertStatement '
    elif ast_type == javalang.tree.AnnotationMethod:
        code_structure += 'AnnotationMethod '
    else:
        if ast_type != list and ast_type != javalang.tree.Statement:
            with open('./other_statement.txt', 'w', encoding='UTF-8') as f:
                f.write(str(ast) + '\n')
                f.write(str(ast_type) + '\n')


def code2ast(code):
    global code_structure
    code_structure = ''
    ast = get_ast(code)
    get_structure(ast)
    return code_structure.split()

if __name__ == '__main__':
    code = '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
    }
    }'''
    print(code2ast(code))

    """
    ClassDeclaration MethodDeclaration StatementExpression 
    """