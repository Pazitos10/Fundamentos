function setup() {
    function checkResults(){
        var hay_results = $.cookie('hay_results');
        if(Boolean(hay_results)){
            cargarTablaResults();
        }
        else
            remove_cookies();
    }

    function cargarTablaResults () {
        $("#page-content").load("resultados_cargados.html");
    }

    function createCanvas(parent, width, height) {
        var canvas = {};
        canvas.node = document.createElement('canvas');
        canvas.context = canvas.node.getContext('2d');
        canvas.node.width = width || 450;
        canvas.node.height = height || 450;
        canvas.node.id = "real-canvas"
        parent.appendChild(canvas.node);
        return canvas;
    }

    function init(container, width, height, fillColor) {
        var canvas = createCanvas(container, width, height);
        var ctx = canvas.context;
        $.cookie("hay_datos", 'False', { expires: null, path: '/'});
        ctx.fillCircle = function(x, y, radius, fillColor) {
            this.fillStyle = fillColor;
            this.beginPath();
            this.moveTo(x, y);
            this.rect(x-15, y-15, radius, radius);
            this.fill();
            $.cookie("hay_datos", 'True', { expires: null, path: '/'}); 
        };
        ctx.clearTo = function(fillColor) {
            ctx.fillStyle = fillColor;
            ctx.fillRect(0, 0, width, height);
        };
        ctx.clearTo(fillColor || "#ffffff");

        // bind mouse events
        canvas.node.onmousemove = function(e) {
            if (!canvas.isDrawing) {
               return;
            }
            var x = e.pageX - this.offsetLeft;
            var y = e.pageY - this.offsetTop;
            var radius = 60;
            var fillColor = '#000000';
            ctx.fillCircle(x, y, radius, fillColor);
        };
        canvas.node.onmousedown = function(e) {
            canvas.isDrawing = true;
        };
        canvas.node.onmouseup = function(e) {
            canvas.isDrawing = false;
        };
    }

    
    var container = document.getElementById('canvas');
    init(container, 450, 450, '#ffffff');
    checkResults();
};


function reconocer(imagen) {
    var datos = imagen;
    $("#matriz_canvas").val(datos);
    $("#form-cgi").submit();
};


function limpiarCanvas(){
    remove_cookies();
    $("#tabla-results").hide();
    var ctx = $('canvas')[0].getContext("2d");
    ctx.clearTo("#ffffff");
    
};


function remove_cookies() {
    var cookies = $.cookie();
    for(var cookie in cookies) {
       $.removeCookie(cookie);
    }
};


function atras(){
    $.cookie('hay_results','',{path:'/', expires:null});
    location.reload();
};