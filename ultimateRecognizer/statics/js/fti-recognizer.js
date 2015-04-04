function setup() {
    // Creates a new canvas element and appends it as a child
    // to the parent element, and returns the reference to
    // the newly created canvas element
    function checkResults(){
        var hay_results = $.cookie('hay_results');
        if(Boolean(hay_results)){
            cargarTablaResults();
        }
        else
            remove_cookies();
    }

    function cargarTablaResults () {
        //$( "#tabla-results" ).load( "resultados_cargados.html" );
        //$("#tabla-results").show();
        $("#page-content").load("resultados_cargados.html");
    }

    function createCanvas(parent, width, height) {
        var canvas = {};
        canvas.node = document.createElement('canvas');
        canvas.context = canvas.node.getContext('2d');
        canvas.node.width = width || 380;
        canvas.node.height = height || 380;
        canvas.node.id = "real-canvas"
        parent.appendChild(canvas.node);
        return canvas;
    }

    function init(container, width, height, fillColor) {
        var canvas = createCanvas(container, width, height);
        var ctx = canvas.context;
        $.cookie("hay_datos", 'False', { expires: null, path: '/'});
        // define a custom fillCircle method
        ctx.fillCircle = function(x, y, radius, fillColor) {
            this.fillStyle = fillColor;
            this.beginPath();
            this.moveTo(x, y);
            //this.arc(x, y, radius, 0, Math.PI * 2, false);
            this.rect(x-15, y-15, radius, radius);
            this.fill();
            $.cookie("hay_datos", 'True', { expires: null, path: '/'}); 
        };
        ctx.clearTo = function(fillColor) {
            ctx.fillStyle = fillColor;
            ctx.fillRect(0, 0, width, height);
        };
        ctx.clearTo(fillColor || "#fefefe");
        //ctx.clearTo(fillColor || "#dddddd");

        // bind mouse events
        canvas.node.onmousemove = function(e) {
            if (!canvas.isDrawing) {
               return;
            }
            var x = e.pageX - this.offsetLeft;
            var y = e.pageY - this.offsetTop;
            var radius = 25;
            //var fillColor = '#0c84e4';
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
    init(container, 380, 380, '#fefefe');
    //init(container, 380, 380, '#dddddd');
    checkResults();

    $("#btn-limpiar").bind("click",limpiarCanvas());

    $("#btn-reconocer").bind("click",function(event) {
        //var ctx = $('canvas')[0].getContext("2d");
        //var datos = ctx.getImageData(0,0,380,380).data
        //var dat = [];
        // for(var i = 0; i < datos.length; i += 1) {
        //     dat.push(datos[i])
        // }
        var datos = $('canvas')[0].toDataURL();
        $("#matriz_canvas").val(datos);
        //console.log($("#matriz_canvas").val());
        $("#form-cgi").submit();
    });
};

function limpiarCanvas(){
    remove_cookies();
    $("#tabla-results").hide();
    var ctx = $('canvas')[0].getContext("2d");
    ctx.clearTo("#fefefe");
    
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