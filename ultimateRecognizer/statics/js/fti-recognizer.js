function setup() {
    // Creates a new canvas element and appends it as a child
    // to the parent element, and returns the reference to
    // the newly created canvas element
    function checkResults(){
        var hay_results = $.cookie('hay_results');
        if(Boolean(hay_results))
            cargarTablaResults();
    }

    function cargarTablaResults () {
        $( "#tabla-results" ).load( "tabla_results.html" );
    }

    function createCanvas(parent, width, height) {
        var canvas = {};
        canvas.node = document.createElement('canvas');
        canvas.context = canvas.node.getContext('2d');
        canvas.node.width = width || 100;
        canvas.node.height = height || 100;
        parent.appendChild(canvas.node);
        return canvas;
    }

    function init(container, width, height, fillColor) {
        var canvas = createCanvas(container, width, height);
        var ctx = canvas.context;
        // define a custom fillCircle method
        ctx.fillCircle = function(x, y, radius, fillColor) {
            this.fillStyle = fillColor;
            this.beginPath();
            this.moveTo(x, y);
            this.arc(x, y, radius, 0, Math.PI * 2, false);
            this.fill();
        };
        ctx.clearTo = function(fillColor) {
            ctx.fillStyle = fillColor;
            ctx.fillRect(0, 0, width, height);
        };
        ctx.clearTo(fillColor || "#ddd");

        // bind mouse events
        canvas.node.onmousemove = function(e) {
            if (!canvas.isDrawing) {
               return;
            }
            $.cookie("hay_datos", 'True', { expires: null, path: '/'}); 
            var x = e.pageX - this.offsetLeft;
            var y = e.pageY - this.offsetTop;
            var radius = 8;
            var fillColor = '#0c84e4';
            ctx.fillCircle(x, y, radius, fillColor);
        };
        canvas.node.onmousedown = function(e) {
            canvas.isDrawing = true;
        };
        canvas.node.onmouseup = function(e) {
            canvas.isDrawing = false;
        };
    }

    $.removeCookie("hay_datos",{ expires: null, path: '/'});
    var container = document.getElementById('canvas');
    init(container, 380, 380, '#ddd');
    checkResults();

    $("#btn-limpiar").bind("click",function(){
        var ctx = $('canvas')[0].getContext("2d");
        ctx.clearRect ( 0 , 0 , $('canvas')[0].width, $('canvas')[0].height );
    });

    $("#btn-reconocer").bind("click",function(event) {
        var ctx = $('canvas')[0].getContext("2d");
        var datos = ctx.getImageData(0,0,200,200).data
        var dat = [];
        for(var i = 0; i < datos.length; i += 1) {
            dat.push(datos[i])
        }
        datos = $('canvas')[0].toDataURL();
        $("#matriz_canvas").val(datos);
        console.log($("#matriz_canvas").val());
        $("#form-cgi").submit();
    });


}